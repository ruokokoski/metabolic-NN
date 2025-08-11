```mermaid
flowchart TD
  subgraph Input
    style Input fill:#ffcccc,stroke:#333,stroke-width:1.5px
    C["c (Batch, Seq, 1)"]
    Tokens["Token idx (0..vocab_size-1)"]
  end

  C -->|input c| Tokens
  Tokens -->|Embeddings| Embedding["x (Batch, Seq, d_model)"]
  style Embedding fill:#ffd9d9,stroke:#333,stroke-width:1.5px

  Embedding --> LN
  Split --> OutputC
  Split --> OutputX

  subgraph LayerCountSubgraph["× L"]
    style LayerCountSubgraph fill:none,stroke:none,font-size:16px,font-weight:bold
  end

  LayerCountSubgraph -.-> FluxTransformerLayer

  subgraph FluxTransformerLayer
    style FluxTransformerLayer fill:#eeeeee,stroke:#333,stroke-width:1.5px
    direction LR
    
    subgraph AttentionBlock["AttentionBlock"]
      style AttentionBlock fill:#ffdd99,stroke:#333,stroke-width:1.5px
      LN["LayerNorm"]
      MHA["MHA"]
      ResidualX["x + attn_out"]
      DiffuseC["Diffuse c via weights"]
      ResidualC["c + c_att"]
    end

    subgraph FeedForwardBlock["FeedForwardBlock"]
      style FeedForwardBlock fill:#cce5ff,stroke:#333,stroke-width:1.5px
      Cat["Concat x_out & c_out"]
      LN2["LayerNorm"]
      FFN["Linear → GELU → Dropout → Linear"]
      ResidualFFN["Add"]
      Split["Split x, c"]
    end
  end

  LayerCountSubgraph -.-> FluxTransformerLayer

  %% AttentionBlock internals
  LN -- Q --> MHA
  LN -- K --> MHA
  LN -- V --> MHA
  MHA --> ResidualX
  MHA --> DiffuseC
  DiffuseC --> ResidualC

  %% Residual connections from inputs to residual nodes in AttentionBlock
  Embedding -- residual --> ResidualX
  C -- residual --> ResidualC

  %% AttentionBlock outputs — separate arrows for x_out and c_out
  ResidualX -->|x_out| Cat
  ResidualC -->|c_out| Cat

  %% FeedForwardBlock internals
  Cat --> LN2
  LN2 --> FFN
  FFN --> ResidualFFN
  ResidualFFN --> Split

  %% Residual connection for FFN block from concatenated input
  Cat -- residual --> ResidualFFN
