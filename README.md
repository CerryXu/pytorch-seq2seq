# pytorch-seq2seq
In this third notebook on sequence-to-sequence models using PyTorch and TorchText, we'll be implementing the model from Neural Machine Translation by Jointly Learning to Align and Translate.

## abstract
In this third notebook on sequence-to-sequence models using PyTorch and TorchText, we'll be implementing the model from [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473).

## The project structure
* pytorch-seq2seq:
    - .data 
    >- multi30k   
    - model
    >>- attention.py    
    >>- decoder.py    
    >>- encoder.py    
    >>- seq2seq.py    
    - train.py
    
## Preparing Data
`train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'), 
                                                    fields = (SRC, TRG))`
The training data is multi30k                                                    

                                                    
## Building the Seq2Seq Model

### Encoder
As we want our model to look back over the whole of the source sentence we return outputs, the stacked forward and backward hidden states for every token in the source sentence. We also return hidden, which acts as our initial hidden state in the decoder.

### Attention
Next up is the attention layer. This will take in the previous hidden state of the decoder

### Decoder
The decoder contains the attention layer, attention, which takes the previous hidden state, s_t-1, all of the encoder hidden states, H, and returns the attention vector, a_t.

We then use this attention vector to create a weighted source vector, w_t, denoted by weighted, which is a weighted sum of the encoder hidden states, H, using a_t as the weights.

The input word (that has been embedded),y_t, the weighted source vector, w_t, and the previous decoder hidden state, s_{t-1}, are then all passed into the decoder RNN, with $y_t$ and w_t being concatenated together.


We then pass y_t, w_t and s_t through the linear layer, f, to make a prediction of the next word in the target sentence. This is done by concatenating them all together.

The image below shows decoding the first word in an example translation.

### Seq2Seq
This is the first model where we don't have to have the encoder RNN and decoder RNN have the same hidden dimensions, however the encoder has to be bidirectional. This requirement can be removed by changing all occurences of enc_dim * 2 to enc_dim * 2 if encoder_is_bidirectional else enc_dim.

This seq2seq encapsulator is similar to the last two. The only difference is that the encoder returns both the final hidden state (which is the final hidden state from both the forward and backward encoder RNNs passed through a linear layer) to be used as the initial hidden state for the decoder, as well as every hidden state (which are the forward and backward hidden states stacked on top of each other). We also need to ensure that hidden and encoder_outputs are passed to the decoder.

## conclusion
1. This project implemented a Seq2Seq based model for Attention translation tasks

2. This project used multi30k as the original data and trained for 20 times

3. Training methods can be added in the later stage of machine translation, and other methods will be gradually tested in the later stage.