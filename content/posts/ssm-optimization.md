---
title: "SSM Optimization: An introduction to GPU performance optimization through the lens of a single kernel"
date: 2023-01-17T04:20:46-08:00
draft: false
---

[I](https://twitter.com/cis_female) didn't have a job and was jonesing for some fun. [A friend](https://twitter.com/typedfemale) was working on a new machine learning architecture called State Space Models (SSMs). What better use of my time than being her personal performance engineer<!-- and peering underneath the GPU's skirt? -->? We got cracking at [Noisebridge](https://noisebridge.net/), and in the course of a few amphetamine-fueled hours, we had a should-be-functional [Triton](https://github.com/openai/triton) kernel. Unfortunately our code was so perfect we crashed the compiler in [two](https://github.com/openai/triton/issues/639) [different](https://github.com/openai/triton/issues/640) ways. Still amphetamine-fueled but now disgusted with Triton and determined to Go Deeper, I went home and read the entire [PTX ISA documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html). The next day I started working on a PTX kernel for MIMO state space models. It took two weeks of scribbling about in-register transposes on a notebook and was deemed "[psychotic](https://twitter.com/typedfemale/status/1571025861155127296)" by fans. But it was useless -- my scribbles went in the trash, my psychotic creation crushed by reality's Risperdal.



<!-- TODO -- improve this explanation. Also -- where do I put this relative to the explanation of what diagonalized SSMS are. -->
I personally tend to eschew intuitive explanations of architectures and am instead attracted to more [mechanistic explanations](https://blog.nelhage.com/post/transformers-for-software-engineers/) of what the underlying math operation is. For State Space Models, the math is that you have some single-dimensional input array `u` of length `SEQUENCE_LENGTH` and a single-dimensional output array `y` of the same size. At every time step, you have a state vector, X, of size `STATE_SIZE`. A typical size for `STATE_SIZE` is e.g. 32 or 64. You have three parameters you try to fit: 1) a state transition matrix `A` of size (`STATE_SIZE`, `STATE_SIZE`), 2) a "new information introduction vector" `B` of size `STATE_SIZE`, and 3) a "state to output" vector `C` also of size `STATE_SIZE`. You then use these parameters as such to calculate your outputs: Xₜ = AXₜ₋₁ + Buₜ, yₜ=CXₜ . Note that this is totally linear. The vast majority of the work in these systems comes from multiplying A by X, because those are much bigger <!-- TODO: improve this sentence... -->. Dealing with this has been *the* major difficulty in making SSMs work practically.

<!-- TODO: explain heads -->


A few months after the crushing defeat of MIMO, the [Friend](https://en.wikipedia.org/wiki/Public_Universal_Friend) started to get into a different way to train SSMs where you [diagonalize](https://mathworld.wolfram.com/MatrixDiagonalization.html) the A matrix. This reduces the computational requirements dramatically, from O(STATE_SIZE^2) flops to O(STATE_SIZE) flops [^1]. There are a lot of questions about how numerically stable this is and whether it can really learn. But it's a fun challenge for optimization, because the math operation we're doing is now quite simple: For each head, take in three vectors: A, B, C, all of the same time. At every timestep, multiply your state by A, load in an input, multiply the input by B, add the state and the input together, multiply the resulting state by C, then write out the output.

So how fast can we make it? How long must we think to achieve optimal performance?

## Torch

A natural first choice to implement any machine learning algorithm is Pytorch. Here's a simple Pytorch function that implements our mathematical operation.

<!-- TODO: make sure all further comments match this format, to have consistent mappings between math ops and gpu ops. -->

```python
@torch.jit.script
def torch_diag(sequence, A, B, C):
    # get our shapes from tensors passed in
    N_HEADS, SEQUENCE_LENGTH = sequence.shape
    STATE_SIZE = B.shape[1]
    # allocate our output
    torch_diag_outputs = torch.empty((N_HEADS, SEQUENCE_LENGTH), dtype=sequence.dtype, device=sequence.device)
    for i in range(N_HEADS):
        # create our state
        state = torch.zeros((STATE_SIZE,), dtype=torch.float32, device=sequence.device)
        for j in range(SEQUENCE_LENGTH):
            previous_state_multed = A[i] * state # multiply previous state by A
            input_by_b = B[i] * sequence[i][j] # multiply new input by B
            state = previous_state_multed + input_by_b # add them together to get new state
            state_by_c = C[i] * state # multiply new state by C
            summed = torch.sum(state_by_c) # sum the result
            torch_diag_outputs[(i, j)] = summed # write to output buffer
    return torch_diag_outputs
```

Fairly intuitive -- so how does it perform?

<!-- include benchmark text from triton with torch results -->

Fucking awful! This function can process 50,000 elements[^2]/second [^3], which is slower than even a single CPU core. Why?
It launches a separate kernel and waits for it to complete for every operation, where it does only `STATE_SIZE` (32) flops. This means that almost all of the power of the GPU is wasted on overhead. I never thought pure Torch would be performant [^4], though, so I didn't spend much time on this and moved quickly to Triton [^5].

## Triton
I have incredibly mixed feelings about Triton. When it works, it's magical. You write Torch-like vector/matrix operations and with only academic knowledge of the GPU even Research Scientists can produce highly performant code. The struggle of Triton is the compiler. Every other time I have tried to write Triton code I've spent 1 hour writing the kernel and 10 hours debugging segfaults in the compiler. Even the [language tutorials](https://github.com/openai/triton/blob/master/python/tutorials/06-fused-attention.py#L18) have comments describing how they got around compiler bugs. The raison d'être for Triton's existence, in my opinion, is efficient code generation with matmuls, because matmuls and tensor cores break the fundamental abstraction of CUDA, the thread. Even though we're not using matmuls here, Triton is still quite good. Let's look at the code I wrote for the ssm_kernel:

// TODO: my triton code is wrong somehow, need to fix it. Need to make sure it produces correct values.

<!-- TODO: explain how A/B/C are math matrices but gpu vectors -->

```python
@triton.jit
def ssm_kernel(sequence_ptr, a_ptr, b_ptr, c_ptr, output_ptr,
               SEQUENCE_LENGTH: tl.constexpr, STATE_SIZE: tl.constexpr,
               N_HEADS: tl.constexpr):
    head = tl.program_id(axis=0) # which head we're on
    # Load A, B, C vectors for this head. they will be held in registers. 
    # i * STATE_SIZE + tl.arange(0, STATE_SIZE) means "load the STATE_SIZE values after index i * STATE_SIZE".
    A = tl.load(a_ptr + head * STATE_SIZE + tl.arange(0, STATE_SIZE))
    B = tl.load(b_ptr + head * STATE_SIZE + tl.arange(0, STATE_SIZE))
    C = tl.load(c_ptr + head * STATE_SIZE + tl.arange(0, STATE_SIZE))
    # create our state vector. this will be held in registers as well.
    state = tl.zeros((STATE_SIZE,), dtype=tl.float32)
    for i in range(SEQUENCE_LENGTH):
        # calculate what index the `i`th value has has for head `head`
        idx = (head * SEQUENCE_LENGTH + i)
	    # load input value
        input_value = tl.load(sequence_ptr + idx)
        # multiply previous state by A, new input by B, add them together
        state = state * A + B * input_value
        # multiply new state by C and sum result
        output_value = tl.sum(state*C, axis=0)
        # write to output buffer
        tl.store(output_ptr + idx, output_value)
```

This looks similar to Torch, but with the added complexity of thinking directly about memory loads and stores as opposed to just indexing into a matrix. However, it blows Torch out of the water on speed: !!4,600,000,000!! elements/s. Why is this? For one thing, it sends just one massive operation to the GPU instead of `5 * SEQUENCE_LENGTH * N_HEADS` operations. How do we understand what's actually going on inside this thing though? Just look at [the assembly](https://godbolt.org/z/4dT54Ejhd)!

A sidenote -- many are scared of assembly, but on GPUs you generally write orders of magnitude less code and spend orders of magnitude more money on running it, so diving into the assembly is often worthwhile. So be courageous, and dive a level deeper with me!

First a quick digression -- what is assembly on GPUs? By "assembly" here I mean [PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) which is a language NVIDIA created as a lower level than CUDA so others writing high-level languages (e.g. [Google's CUDA compiler](https://research.google/pubs/pub45226/), [Halide](https://github.com/halide/Halide), Triton) could have a low-level language to compile to. /* This is sort of a bizarre situation, like if Intel had created hardware to run C on and then created assembly after the fact. TODO: include this?? */. /* Below PTX there's another language called "SASS" which is like GPU microcode -- the hardware executes it directly and it changes from generation to generation. NVIDIA really doesn't want you to use this, because if you write SASS it won't work with new GPUs. But you can do it anyway :p. TODO: include this? */ If you want to understand what Triton is actually doing, you have to look at the PTX. So off we go.

The inner loop of our compiled kernel looks like this:
```c
LBB0_1:
	ld.global.b32 {%r13}, [ %rd21 + 0]; // load input value to %r13
	mov.b32  %f17, %r13; // move the value in %r13 (generic 32 bit register) to %f17 (floating point register).
	// this does nothing physically.

    # multiply previous state by A, new input by B, add them together
	mul.f32  %f18, %f2, %f17; // multiply new input (%f17) by B (%f2) -> %f18
	fma.rn.f32  %f20, %f20, %f1, %f18; // Multiply previous state (%f20) by A (%f1) and then add this to (new_input * B) (%f18)

	// calculate pointer to write to. Why does it do this in the inner loop?
	// Your guess is as good as mine.
	mul.wide.s32  %rd20, %r20, 4; // multiply head we're in by 4 and put it in %rd20
	add.s64  %rd19, %rd4, %rd20; // add %rd20 (head * 4) to output ptr

	mul.f32  %f8, %f20, %f3; // X (%f20) * C (%f3) -> %f8

	// Five butterfly shuffles. Butterfly shuffles need a diagram to explain, so see the main text.
	shfl.sync.bfly.b32  %f7, %f8, 16, 0x1f, 0xffffffff;
	fma.rn.f32  %f10, %f20, %f3, %f7; // here it recalculates %f8 (%f20 * %f3) for some reason
	shfl.sync.bfly.b32  %f9, %f10, 8, 0x1f, 0xffffffff;
	add.f32  %f12, %f10, %f9;
	shfl.sync.bfly.b32  %f11, %f12, 4, 0x1f, 0xffffffff;
	add.f32  %f14, %f12, %f11;
	shfl.sync.bfly.b32  %f13, %f14, 2, 0x1f, 0xffffffff;
	add.f32  %f16, %f14, %f13;
	shfl.sync.bfly.b32  %f15, %f16, 1, 0x1f, 0xffffffff;
	add.f32  %f19, %f16, %f15;
	mov.b32  %r19, %f19; // completed calculating butterfly shuffles

	st.global.b32 [ %rd19 + 0] , {%r19}; // write out output

	add.s32  %r21, %r21, 1; // increment sequence index
	add.s32  %r20, %r20, 16; // add 16 to the output ptr. ??? why 16
	add.s64  %rd21, %rd21, 64; // increment u_ptr by 16
	setp.lt.s32  %p6, %r21, 8192;
	@%p6 bra LBB0_1;
	ret;
```

If we compare to our original code, we can clearly see each line of code corresponding to one or two lines of assembly. Great! I'm starting to feel like PTX is a breeze! But then there's this big block of shuffles. Now we have to start thinking about "warps" and "threads" so get ready.


crappy butterfly shuffle diagram, will make my own.
![Data exchange among 8 threads using butterfly warp shuffle operations. |  Download Scientific Diagram](https://www.researchgate.net/publication/317485271/figure/fig1/AS:505251083100160@1497472653903/Data-exchange-among-8-threads-using-butterfly-warp-shuffle-operations.png)

why is [SSM kernel with STATE_SIZE=4096](https://godbolt.org/z/KbGTxb4oh) 60x slower than [SSM kernel with STATE_SIZE=2048](https://godbolt.org/z/KbGTxb4oh)? challenge for the devoted reader! <!-- if you remove the 1-warp requirement it's only twice as slow, which we would expect. my guess is it hits some kind of limit for the number of registers in a SMSP. -->

<!-- TODO: mathml/latec? -->
[^1]: Specifically from 2 * STATE_SIZE^2 + 4 * STATE_SIZE flops to 5 * STATE_SIZE flops. For STATE_SIZE=32 this is a 13x decrease (2176 flops to 160 flops)
[^2]: An "element" in my calculation is one value for one head, so e.g. 32 heads with 512 sequence length each is 16,384 "elements"
[^3]: All benchmarks are performed on an A100 and with `STATE_SIZE`=32 unless otherwise stated.
[^4]: To be fair, this could be much faster than it is currently. [Horace He](https://twitter.com/chhillee) recommended that batching over heads would yield significant improvements, because there would be `N_HEADS` fewer kernel launches. I'm not doing this to be practical though -- I'm doing it to be OPTIMAL
[^5]: Given this, you might reasonably ask why I would ever write the Torch version in the first place: because it's much easier to debug and understand. All other versions of this function are tested against the Torch version to ensure they return similar results.