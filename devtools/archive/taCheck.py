import py_compile
import pathlib

base_path = pathlib.Path("../patchedLibs/pandas-ta-main-local/pandas_ta")
errors = []

# Recursively compile all .py files
for py_file in base_path.rglob("*.py"):
    try:
        py_compile.compile(str(py_file), doraise=True)
        print(f"[OK]     {py_file.relative_to(base_path)}")
    except py_compile.PyCompileError as e:
        print(f"[ERROR]  {py_file.relative_to(base_path)}")
        errors.append((py_file, e))

print("\n=== Summary ===")
print(f"✅ Compiled OK: {len(list(base_path.rglob('*.py'))) - len(errors)}")
print(f"❌ Errors:      {len(errors)}")

if errors:
    print("\n--- Files with Errors ---")
    for file, err in errors:
        print(f"{file.relative_to(base_path)}\n  {err}")


