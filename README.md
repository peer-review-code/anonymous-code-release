# Code release

For the final paper we will improve the documentation and provide the scripts used to generate the results. This is the main code setup for the paper, with `main.py` providing all the interfaces. `data` contains the different datasets preprocessed for our analysis.

# Setup


## Install Package

Clone the repository and install the package using the following command:

```
pip install --editable . # install the package in editable mode since it is still under development
```

This should install the package and its dependencies. The install is in editable mode so that if you change the code you don't need to reinstall the package.

## Run Tests

`/data` contains sample input and output files for testing the code. The current versions to target are `sample_script_input_new.json` and `sample_script_output_new.json`.

To run the tests, use the following command:

```
python main.py --infile data/sample_script_input_new.json --outfile test.json
```

This will generate a file `test.json` with the output of the model. The output should match the `sample_script_output_new.json` file exactly.

# Usage

The main usage is via `main.py`. The script takes in an input json file and generates an output line json file with one line per prompt. The input json file should have the following format, note that delimiters are required to denote which substrings of the input prompt are processed in parallel:

```
{
    {
            "prompt": "A <|start_2d|>B<|split_2d|>C<|split_2d|>D<|end_2d|>E"
    },
    {
            "prompt": "F <|start_2d|>G<|split_2d|>H<|split_2d|>I<|end_2d|>J"
    }
}
```

`<|start_2d|>` indicates the start of a parallel processing block, `<|split_2d|>` indicates the split between parallel processing blocks, and `<|end_2d|>` indicates the end of a parallel processing block. The script only does a basic check to make sure the delimiters are present and in the correct order, but does not check for any other errors.

The output json file will look something like this:

```
{
  "prompt": "What animal makes the best friendly, outgoing pet? Options: <|start_2d|>dog<|split_2d|>cat <|split_2d|>hamster<|end_2d|>. Answer: ",
  "output_order_dependent": {
    "prompt": "What animal makes the best friendly, outgoing pet? Options: <|start_2d|>dog<|split_2d|>cat <|split_2d|>hamster<|end_2d|>. Answer: ",
    "model": "GPT2LMHeadModel",
    "max_new_tokens": 10,
    "order_independent_output": false,
    "pad_attention": false,
    "text_output": "dogcat is a great pet for your dog."
  },
  "output_order_dependent_rev": {
    "prompt": "What animal makes the best friendly, outgoing pet? Options: <|start_2d|>hamster<|split_2d|>cat <|split_2d|>dog<|end_2d|>. Answer: ",
    "model": "GPT2LMHeadModel",
    "max_new_tokens": 10,
    "order_independent_output": false,
    "pad_attention": false,
    "text_output": " \"I think it's a cat. I"
  },
  "output_order_independent": {
    "prompt": "What animal makes the best friendly, outgoing pet? Options: <|start_2d|>dog<|split_2d|>cat <|split_2d|>hamster<|end_2d|>. Answer: ",
    "model": "GPT2LMHeadModel",
    "max_new_tokens": 10,
    "order_independent_output": true,
    "pad_attention": false,
    "text_output": "cat................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................"
  }
}
```

This is for the three different outputs: the output of the model with the original order, the output of the model with the order of the parallel substrings reversed, and the output of the model with the parallel substrings processed in parallel.


# Previous version of the README


## Generate order independent output for a given model and input prompts
The following command takes in a json file of input prompts, and produces a json file containing the model outputs, as produced by either an order dependent (default implementation model), or an order independent (intervention) model.

Supported options for model-name are ["gpt2","meta-llama/Llama-2-7b-chat-hf","meta-llama/Llama-2-7b-hf"].
```
python gen_order_independent_output.py --model-name gpt2 --torch-device cpu --max-new-tokens 10 --infile path/to/input.json --outfile path/to/output.json
```
See `data/sample_script_input.json` and `data/sample_script_output.json` for sample input and output files. Use `<|start_2d|>` `<|split_2d|>`, `<|end_2d|>` strings as delimiters to denote which substrings of the input prompt are processed in parallel.

E.g. a sample prompt might have the form "prompt":"A <|start_2d|>B<|split_2d|>C<|split_2d|>D<|end_2d|>E", where B,C,D are substrings processed in parallel.

Output format:
Each output entry contains the keys: "prompt" for the original prompt, "output_order_dependent" for the output generated by the baseline model, "output_order_dependent_rev" for the output generated by the baseline model when the order of the parallel substrings is reversed (e.g. A(DCB)E instead of A(BCD)E), and "output_order_independent" for the output text generated when the substrings are processed in parallel.
