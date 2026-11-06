"""Derive publishable AST symbol facts without literal/code-bearing receivers.

The original observer stdout stays unchanged in the controller ledger.
This transforms operational observations only, never scientific result JSON.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def symbol(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return symbol(node.value) + '.' + node.attr
    if isinstance(node, ast.Call):
        return 'call(' + symbol(node.func) + ')'
    if isinstance(node, ast.Constant):
        return '<literal:' + type(node.value).__name__ + '>'
    return '<' + type(node).__name__ + '>'


def derive(raw):
    value = json.loads(raw)
    changes = []

    def visit(item, path):
        if isinstance(item, list):
            for i, child in enumerate(item):
                visit(child, path + '/' + str(i))
        elif isinstance(item, dict):
            if 'callee' in item:
                original = item['callee']
                node = ast.parse(original, mode='eval').body
                replacement = symbol(node)
                if original != replacement:
                    entry = {'json_path': path + '/callee',
                             'original_expression_sha256': digest(original.encode()),
                             'original_expression_bytes': len(original.encode()),
                             'symbolic_callee': replacement}
                    modules = sorted({n.args[0].value for n in ast.walk(node)
                        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                        and n.func.id == '__import__' and n.args
                        and isinstance(n.args[0], ast.Constant)
                        and isinstance(n.args[0].value, str)
                        and re.fullmatch(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*', n.args[0].value)})
                    if modules:
                        entry['dynamic_import_module_names'] = modules
                    item['callee'] = replacement
                    changes.append(entry)
            for key, child in item.items():
                visit(child, path + '/' + key)

    visit(value, '')
    value['publication_derivation'] = {
        'kind': 'sanitized_derivative_not_byte_exact_original_stdout',
        'original_stdout_sha256': digest(raw),
        'original_stdout_bytes': len(raw),
        'rule': 'AST callee expressions reduced to symbol/node kinds; arguments and literal receivers omitted',
        'changes': changes,
    }
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = derive(args.input.read_bytes())
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print('Derived', len(result['publication_derivation']['changes']), 'symbol-only callee records')
