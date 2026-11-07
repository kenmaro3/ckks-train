"""Verify every existing test body token, including assertions and literal arguments."""
import pathlib,subprocess,json,re
l=pathlib.Path(__file__).parent
root=pathlib.Path(__file__).resolve().parents[3]
base='ec2d2656f21e9139d3dde12abdce13d770ea2757'
rows=[]
# Lex strings/raw strings before discarding Rust whitespace and comments.
pattern=re.compile(r'//[^\n]*|/\*.*?\*/|r(#+)?".*?"\1|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])\'|[^\s]',re.S)
def tokens(text):
    result=[m.group() for m in pattern.finditer(text) if not m.group().startswith(('//','/*'))]
    # Rustfmt may add/remove trailing delimiter commas, without changing a value.
    return [v for i,v in enumerate(result) if not (v==',' and i+1<len(result) and result[i+1] in (')',']','}'))]
for name in ['case_provenance_tests','execution_binding_tests','target_link_tests','target_tests','evidence_tests','admission_tests']:
    p=pathlib.Path(f'crates/ckks-train-bench/src/study_reference/document/{name}.rs')
    old=subprocess.check_output(['git','show',f'{base}:{p}'],cwd=root).decode()
    new=(root/p).read_text()
    expected=old[old.index('#[test]'):]
    for ident in ['LOCK','REFERENCE_BYTES','INITIAL','RETRY_NONE','RETRY_SOME','START_INITIAL','START_RETRY']:
        expected=re.sub(r'\b'+ident+r'\b','literal_'+ident.lower()+'()',expected)
    expected=re.sub(r'include_bytes!\(\s*"../target_goldens/([^\"]+)"\s*\)(?:\.as_slice\(\))?',lambda m:f'target_literal("{m[1]}").as_bytes()',expected)
    expected=re.sub(r'include_str!\(\s*"../target_goldens/([^\"]+)"\s*\)',lambda m:f'target_literal("{m[1]}")',expected)
    assert tokens(expected)==tokens(new[new.index('#[test]'):]),name
    assertions=len(re.findall(r'\bassert(?:_eq|_ne)?!',old))
    assert assertions==len(re.findall(r'\bassert(?:_eq|_ne)?!',new))
    rows.append(dict(path=str(p),original_tests=len(re.findall(r'#\[test\]',old)),preserved_assertions=assertions,all_post_first_test_tokens_equal_after_fixture_substitutions=True))
(l/'task-54-d7-fixture-custody-assertion-check.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
