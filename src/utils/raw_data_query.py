RAW_DATA_SQL = """
SELECT 
    a.activity_id,
    a.assay_id,
    a.molregno,
    a.standard_value as IC50,
    a.standard_units,
    cs.canonical_smiles as smiles,
    ass.assay_type,
    ass.confidence_score,
    ass.tid as target_id, 
    td.chembl_id,
    td.target_type,
    td.organism,
    cp.*
FROM activities a
JOIN assays ass ON a.assay_id = ass.assay_id 
JOIN molecule_dictionary md ON a.molregno = md.molregno
JOIN compound_structures cs ON a.molregno = cs.molregno
JOIN target_dictionary td ON ass.tid = td.tid
LEFT JOIN compound_properties cp ON a.molregno = cp.molregno
WHERE 
    standard_type = 'IC50' 
AND cs.canonical_smiles IS NOT NULL 
AND a.standard_value IS NOT NULL
AND a.standard_value > 0
AND a.standard_units IS NOT NULL
AND a.standard_relation = '='
AND a.potential_duplicate = 0
AND md.black_box_warning = 0
"""


def get_data_query() -> str:
    return RAW_DATA_SQL
