# Models

## Naming and Storage standard

Models should be stored as both a serialized object and its `.py` source (as a class or function). Recovery may be accomplished, for instance, through `torch.load`. Serialized objects should have the same name as the `.py` file with `.ckpt` extension. Serialized objects are not source controlled.

