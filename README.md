# Install
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```
## Format
`autopep8 --in-place --aggressive --aggressive query.py`


## Example usage(s)

### look for hlsl intrinsics
```powershell
 python .\agregate_hlsl_intrinsic_usage.py "D:\\projects\\DirectML\\Product\\Shaders" > results.txt
```