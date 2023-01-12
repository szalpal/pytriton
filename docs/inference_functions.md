<!--
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Inference Functions Design

The following section provide more details about writing inference function for Trition.


### Inference Function

The inference function is an entrypoint for inference. The interface of inference function (or callable) assumes
that it receives list of requests as dictionaries (each dictionary represents one request mapping model input names
into numpy ndarrays).

In the simple implementations (e.g. functionality that pass input data on output) the lambda can be used as follows:

```python
triton.bind(
    model_name="Identity",
    infer_func=lambda inputs: inputs,
    inputs=[Tensor(dtype=np.float32)],
    outputs=[Tensor(dtype=np.float32)],
    config=ModelConfig(max_batch_size=8)
)
```

For more complex cases, we suggest creating a separate inference function that can implement more logic of processing
the input data and create output response.

### Batching decorator

In many cases it is much more convenient to get input already batched in form of numpy array
(instead of list of separate requests).
For such cases we prepared `@batch` decorator that adapts generic input into batched form (it passes kwargs to inference
function where each named input contains numpy array with batch of requests received by Triton server).

Below we show difference between decorated and undecorated function bound with triton:

```python
#sample input data with 2 requests - each with 2 inputs
inputs = [{'in1': np.array([[1, 1]]), 'in2': np.array([[2, 2]])},
          {'in1': np.array([[1, 2]]), 'in2': np.array([[2, 3]])}          ]

def undecorated_infer(inputs):
  print(inputs)
  # as expected inputs = [{'in1': np.array([[1, 1]]), 'in2': np.array([[2, 2]])},
  #                      {'in1': np.array([[1, 2]]), 'in2': np.array([[2, 3]])}

@batch
def decorated_infer(in1, in2):
  print(in1, in2)
  # in1 = np.array([[1, 1], [1, 2]])
  # in2 = np.array([[2, 2], [2, 3]])
  # inputs are batched by `@batch` decorator and passed to function as kwargs so can be automatically mapped
  # with in1, in2 function parameters
  # Of course we could get this params explicitly with **kwargs like this:
  #  def decorated_infer(**kwargs):

undecorated_infer(inputs)
decorated_infer(inputs)
```

More examples using `@batch` decorator and different frameworks below.


#### Example implementation for TensorFlow model:

```python
@batch
def infer_func(**inputs: np.ndarray):
    (images_batch,) = inputs.values()
    images_batch_tensor = tf.convert_to_tensor(images_batch)
    output1_batch = model.predict(images_batch_tensor)
    return [output1_batch]
```

#### Example implementation for PyTorch model:

```python
@batch
def infer_func(**inputs: np.ndarray):
    (input1_batch,) = inputs.values()
    input1_batch_tensor = torch.from_numpy(input1_batch).to("cuda")
    output1_batch_tensor = model(input1_batch_tensor)
    output1_batch = output1_batch_tensor.cpu().detach().numpy()
    return [output1_batch]
```

#### Example implementation with [named inputs and outputs](#inputs-and-outputs) `a` and `b`:

```python
@batch
def _add_sum_fn(a: np.ndarray, b: np.ndarray):
    return {"add": a + b, "sub": a - b}

@batch
def _mul_fn(**inputs: np.ndarray):
    a = inputs["a"]
    b = inputs["b"]
    return [a * b]
```

#### Example implementation with strings:

```python
@batch
def _classify_text(text_array: np.ndarray):
    a = text_array[0]  # text_array contains one string at index 0
    a = a.decode('utf-8')  # string is stored in byte array enocded in utf-8
    res = classifier(a)
    return [np.array(res)]  # return statisctics generated by classifer
```

### More decorators adapting input to user needs


We have prepared a few useful decorators for adapting generic request input into common user needs.
It is of course possible to write your own custom decorators that perfectly fit to your needs
and chain them with other decorators.

Our common decorators realize three type of interfaces:
#### a) receive the list of request dictionaries and return the list of response dictionaries (dictionaries map input/output names into numpy arrays)
- `@group_by_keys` - prepare groups of requests with the same set of keys and calls wrapped function
for each group separately (it is convenient to use this decorator before batching, because the batching decorator
requires consistent set of inputs as it stacks them into batches)
```python
@group_by_keys
@batch
def infer(mandatory_input, optional_input=None):
    #do inference
    pass
```
- `@group_by_values(*keys)` - prepare groups of requests with the same input value (for selected keys) and
calls wrapped function for each group separately (it is especially convenient to use with models that
requires dynamic parameters sent by user e.g. temperature - in this case we would like to run model only
for requests with the same temperature value)
```python
@group_by_values('temperature')
@batch
def infer(mandatory_input, temperature):
    #do inference
    pass
```
- `@fill_optionals(**defaults)` - fills missing inputs in requests with default values provided by user (if model owner
have default values for some optional parameter it is good idea to provide them at the very beginning, so
the other decorators can make bigger consistent groups and send them to the inference function)
```python
@fill_optionals(temperature=np.array([10.0]))
@group_by_values('temperature')
@batch
def infer(mandatory_input, temperature):
    #do inference
    pass
```

#### b) receive the list of request dictionaries and return dictionary that maps input names into arrays
(passes dictionary to wrapped infer function as named arguments - kwargs):
- `@batch` - generates batch from input requests
- `@sample` - take first request and convert it into named inputs (useful with non-batching models -
instead of one element list of request, we will get named inputs - `kwargs`)
  (usage is shown in previous examples)

#### c) receive batch (dictionary that maps input names into arrays) and return batch after some processing:
- `@pad_batch` - appends last row to input multiple times to get desired batch size (preferred batch size or
max batch size from model config whatever is closer to current input size)
```python
@batch
@pad_batch
def infer(mandatory_input):
    #this model requires mandatory_input batch to be the size provided in the model config
    pass
```

- `@first_values` - This decorator takes first elements from batches for selected inputs by `keys` parameter.
If the value is one element array, it is converted to scalar value.
It is convenient to use with dynamic model parameters that are sent by the user in the requests.
You can use `@group_by_values` before to have batches with the same values in each batch.
```python
@fill_optionals(temperature=np.array([10.0]))
@group_by_values('temperature')
@batch
@first_values('temperature')
def infer(mandatory_input, temperature):
    #do inference with scalar temperature=10
    pass
```

#### d) There is special decorator called @triton_context
It gives you additional argument called 'triton_context' - you can read model config from it and
in the future possibly have some interaction with triton

```python
@triton_context
def infer(input_list, **kwargs):
    model_config = kwargs['triton_context'].model_config
    #do inference using some information from model_config
    pass
```

***
**At the end we show the example of stacking even more decorators together**
As it is shown below, we have to start with a) type decorators then use b) and c)
We recommend putting `@triton_context` as the last decorator in chain.

```python
@fill_optionals(temperature=np.array([10.0]))
@gorup_by_keys
@group_by_values('temperature')
@batch
@first_values('temperature')
@triton_context
def infer(triton_context, mandatory_input, temperature, opt1 = None, opt2 = None):
    model_config = triton_context.model_config
    #do inference using:
    #   - some information from model_config
    #   - scalar temperature value
    #   - optional parameters opt1 and/or opt2
    pass
```