---
date: '2024-12-04T23:12:44+08:00'
draft: false
math: true
title: 'Reasoning Horizon Scaling - Efficient Architecture (I)'
---


Humans are extraordinary at slow and persistent reasoning. This has not yet been achieved by current LLMs, mainly due to two reasons:  
- Limited memory capacity and awkward length scaling efficiency.  
- High variance in learning signals and sparse rewards.  

I will discuss the first issue in this blog and leave the second one for the future.  

In the first stage, there are primarily two directions: the first is memory augmentation, and the second is efficient transformers. In this blog, I will focus only on the second direction.  

Specifically, this blog will cover the following topics:  
1) Linear Transformer  
2) Hopfield Network  
3) Delta Net
4) Chunk Parallel for GAU
5) A new idea (BSBR)  

---

# Linear Transformer  

Denote the Query, Key, and Value matrices in the transformer as \( Q \), \( K \), \( V \in \mathbb{R}^{n \times d} \).  
The traditional transformer computes the output \( O \) as:  

\[
O = \text{softmax}(QK^T)V
\]

The linear transformer removes the softmax operator, instead computing \( O \) as:  

\[
O = (QK^T)V
\]  

The removal of the softmax operator leverages the associative property of matrix multiplication. Using this property, we can rewrite \( O \) as:  

\[
O = Q(K^TV)
\]  

Here, \( K^TV \in \mathbb{R}^{d \times d} \), which does not expand while scaling with sequence length.  

Additionally, \( K^TV \in \mathbb{R}^{d \times d} \) can be expressed as:  

\[
\sum_{i=1}^{n} k_i^T v_i
\]  

Thus, it has the following recurrent form:  

\[
S_i = S_{i-1} + k_i^T v_i
\]  


---

## Relationship with Hopfield Network  

It is generally acknowledged that the linear transformer has a close relationship with the Hopfield Network. A Hopfield Network \( f_{\{x_i\}}(x') \), trained on a set of inputs \( \{x_i\} \), is considered to have memory capabilities. Specifically, when the network is given a new input \( x' = x_{i0} + \epsilon \), where \( \epsilon \) is small noise, iterating through \( f \), as in \( f(f(...f(x')))\), will retrieve \( x_{i0} \).  

Now, let’s explore why the linear transformer is closely related to the Hopfield Network and what this relationship entails:  

\[
k' S_n k = k' \sum_i k_i^T v_i = \sum_i (k' k_i^T)v_i
\]  

When \( k_i \) is sufficiently linearly independent, this operation effectively extracts \( v_i \).  

Therefore, for the linear transformer to function like a Hopfield Network, \( k_i \) needs to be independent enough from one another.  

---

## Expressiveness  

It is clear that when \( K^T V \) does not expand with sequence length, it cannot preserve increasing amounts of information.  

However, from the perspective of \( (QK^T)V \), even without softmax, \( QK^T \) is still an \( n \times n \) matrix. This can still be interpreted as a metric (inner product) between \( q_i \) and \( k_j \), where \( q_i \) and \( k_j \) are vectors (or \( 1 \times d \) matrices) for each token.  

In fact, without softmax, despite \( QK^T \) being an \( n \times n \) matrix, its rank is no greater than \( d \). Thus, the \( n \times n \) matrix is effectively redundant.  

So, why is softmax so special? It applies \( \exp(\cdot) \), introducing non-linearity into the process.  

This highlights the core drawback of the linear transformer: it lacks expressiveness. Most follow-up works aim to enhance the expressiveness of linear transformers without sacrificing their efficiency.  

---

## DeltaNet

DeltaNet improves upon the linear transformer by introducing a removal component:  

\[
S_i = S_{i-1} - \beta k_i^T v_{\text{old}} + \beta k_i^T v_i
\]  

Here, \( v_{\text{old}} \) is extracted from the previous state \( S_{i-1} \) using the following formula:  

\[
v_{\text{old}} = k_i S_{i-1}
\]  

This rule is clearly better because, if you set \( \beta = 1 \) and use \( k_i \) to extract information from \( S_i \), you will retrieve only \( v_{\text{new}} \). The parameter \( \beta < 1 \) serves as a conservative hyper-parameter, allowing the model to retain part of the old information.  

Does this method increase expressiveness in terms of rank? Unfortunately, no. The bottleneck for the rank of the resulting output matrix remains \( d \).  

---  


### Parallelizing DeltaNet  

This part is implemented in [recent work](https://arxiv.org/abs/2406.06484).

Denote \( u_i = v_i - v_{\text{old}} \).  

Now, we have:  

\[
S_n = S_{n-1} + \beta k_n^T (v_n - v_{\text{old}}) = S_{n-1} + \beta k_n^T u_n
\]  

Note that \( u_n \) depends on \( S_{n-1} \), so this is not a recurrent definition.  

Define \( u_1 = v_1 \), so:  

\[
S_n = \beta \sum_{i=1}^n k_i^T u_i
\]  

Next, write \( u_n \) in terms of \( u_1, \ldots, u_{n-1} \):  

\[
u_n = v_n - v_{\text{old}} = v_n - \beta \sum_{i=1}^{n-1} k_i k_i^T u_i
\]  

However, this still does not yield a truly recurrent form—it merely shifts the recurrence from \( S \) to \( u \).  

To address this, the authors propose using chunk parallelization for \( u_i \).  

---

### Chunk Parallel  

The derivation of the chunk-parallel form for DeltaNet is somewhat complicated, so we start with a simpler example: [GAU](https://arxiv.org/pdf/2202.10447).  

The idea behind chunk parallelization is straightforward.  

As previously discussed, to compute \( QK^T V \), we have two options: \( (QK^T)V \) or \( Q(K^T V) \). While the latter results in a smaller intermediate matrix, different positions’ \( K_{:t}^T V_{:t} \) are actually *superimposed* over one another. This makes it impossible to apply a causal mask and compute \( K_{:t}^T V_{:t} \) for different positions separately.  

Thus, we have three choices:  
1. Use a recurrent form.  
2. Use parallel scan.  
3. Use \( (QK^T)V \).  

The second option (parallel scan) needs to be closely integrated into the training infrastructure, which we will discuss later (not in this blog).  

The first option (recurrent form) cannot be used for the entire sequence because it is computationally slow. Similarly, the third option (\( (QK^T)V \)) is not viable for the entire sequence because it reverts to quadratic attention.  

A practical solution is to split the sequence into chunks. Within each chunk, we use \( (QK^T)V \). Between chunks, we apply the recurrent form.  

---  


# Block Sparse Attention with Block Retrieval (BSBR)

**This part is newly introduced in this blog, not from other's paper**


#### Can we use \( (QK^T)V \) outside the chunk and use the recurrent form inside?

The intuition behind this idea comes from three perspectives:

1. **Short-term information is redundant**: For example, imagine a tokenized video. While short-term information often contains overlap, long-term information is indispensable, and we must ensure it is neither lost to decay nor overwritten entirely.  
2. **Rank and information volume**: From a matrix rank perspective, in the short-term, where the sequence length \( l \) is comparable to \( d \), the information content of the recurrent form is similar to that of the parallel form with a softmax nonlinearity.  
3. **No free lunch**: If the goal is to recall information from \( 1 \times 10^9 \) tokens ago, that information must be stored somewhere. Thus, reducing memory usage significantly is impractical for achieving lifelong memory.  

Next, we discuss a possible approach to implementing \( (QK^T)V \) outside the chunk while using the recurrent form inside, resulting in a slightly reduced memory footprint and a lower computational cost.

---

#### Proposed Chunking Approach  

Suppose the sequence is split into \( C \) chunks, each of length \( B \), and the total sequence length is \( L \).  

At the end of each chunk \( c \), we compute a state \( S_{Bc} \), which is flattened into a vector \( f_c \). Each chunk also has a state meta key (\( \underline{h} \), or hash) \( h_c \), derived from the state, and a state meta query (\( \underline{r} \), or retriever) \( R_c \). We first compute:  

\[
S^{'} = \text{softmax}(RH^T)F
\]  

In vector form:  

\[
S^{'}_{Bc} = \left( \text{softmax}(r_c H_{:c}^T)F_{:c} \right).\text{flatten}()
\]  

Now, for a second-order query \( q \) at position \( Bc + i \):  

- Start from the state \( S^{'}_c \).  
- Add the new states \( K_{Bc:Bc+i}^T V_{Bc:Bc+i} \):  

\[
S^{'}_{Bc+i} = S^{'}_c + K_{Bc:Bc+i}^T V_{Bc:Bc+i}
\]  

The output for \( q_{Bc+i} \) is computed as:  

\[
O = q_{Bc+i} S^{'}_c + q_{Bc+i} K_{Bc:Bc+i}^T V_{Bc:Bc+i}
\]  

However, directly computing \( K_{Bc:Bc+i}^T V_{Bc:Bc+i} \) is not efficient. Instead, we compute:  

\[
O = q_{Bc+i} S^{'}_c + \left( q_{Bc+i} K_{Bc:Bc+i}^T \right) V_{Bc:Bc+i}
\]  

To further enhance this, we can add back the softmax for nonlinearity:  

\[
O = q_{Bc+i} S^{'}_c + \text{softmax}(q_{Bc+i} K_{Bc:Bc+i}^T)V_{Bc:Bc+i}
\]  

Using matrix notation:

For each chunk \( c \), we compute the following:  
- \( K_{Bc:Bc+B} \) and \( V_{Bc:Bc+B} \).  
- A state meta retriever \( r_c \).  
- A state meta hash \( h_c \).  
- The flattened state vector \( f_c = \text{flatten}(K_{Bc:Bc+B}^T V_{Bc:Bc+B}) \).  

The output is given by:  

\[
O = Q \odot \text{softmax}(RH^T \cdot M_{\text{out}})F.\text{repeat}(B) + \text{softmax}(QK^T \cdot M_{\text{in}})V
\]  

Where:  
- \( Q, K, V \in \mathbb{R}^{L \times d_0} \).  
- \( M_{\text{in}} \in \mathbb{R}^{L \times L} \) is a diagonal block mask. 
- \( M_{\text{out}} \in \mathbb{R}^{\frac LB \times \frac LB} \) is a global causal mask. 
- \( R, H \in \mathbb{R}^{L/B \times d_0} \).  
- \( F \in \mathbb{R}^{L/B \times d_0^2} \).  

Here, \( \odot \) represents a row-wise vector product. From a shape perspective, the first term transforms \( [L, d_0] \odot [CB, d_0, d_0] \) into \( [L, d_0] \), where \( CB = L \).  

---

#### Complexity  

Assuming no compression of \( F \in \mathbb{R}^{d_0^2} \), the computational complexity is:  

\[
O(B^2 \cdot L/B \cdot d_0) + O(L^2/B^2 \cdot d_0^2) + O(L \cdot d_0)
\]  

However, \( d_0^2 \) can be reduced, for example, to \( d_1^2 \ll d_0^2 \).

---

#### Practical Considerations  

- **Storage vs. Compute Trade-off**:  
  Hard drives are inexpensive, but computation is costly. Thus, we focus on reducing computation rather than reducing memory (saved information).  

- **Efficient Retrieval**:  
  The \( F \) matrix can be stored on a hard drive and retrieved into the CPU or GPU as needed. For instance, suppose we select the top 5 rows of \( F \) per retrieval. With \( d_1 = 1024 \), 16 heads, and 80 layers (which is sufficient for persistent memory), the retrieval involves:  

  \[
  5 \cdot 16 \cdot 80 \cdot 1024 \cdot 1024 \approx 6\text{ GB}.
  \]  

  SSD(PCIe 5.0) has a speed exceeds 10 GB/s, and generating one token takes ~1–10 ms. Therefore, retrieving every 1024 tokens is entirely feasible.


