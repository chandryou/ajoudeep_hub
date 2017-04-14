/* 

--generating outcome_date(requires modification)
SELECT *,
	CASE 
		WHEN subject_id in 
			(SELECT subject_id FROM @resultsDatabaseSchema.@cohort WHERE cohort_definition_id = @outcome_id)
		THEN 1 
		ELSE 0 
	END as outcome
INTO #cohort
	FROM @resultsDatabaseSchema.@cohort coh
	WHERE cohort_definition_id = @cohort_definition_id
;

*/

SELECT *,
	CASE 
		WHEN subject_id in 
			(SELECT subject_id FROM @resultsDatabaseSchema.@cohort WHERE cohort_definition_id = @outcome_id)
		THEN 1 
		ELSE 0 
	END as outcome
INTO #cohort
	FROM @resultsDatabaseSchema.@cohort coh
	WHERE cohort_definition_id = @cohort_definition_id
;
	  


SELECT DISTINCT con.person_id, X.outcome, per.gender_concept_id, per.year_of_birth, con.condition_start_date, con.condition_concept_id, X.visit_start_date, X.visit_seq, ROW_NUMBER() OVER (PARTITION BY con.visit_occurrence_id ORDER BY con.condition_start_date asc) AS dx_seq
--, rel.concept_id_2
	FROM @cdmDatabaseSchema.CONDITION_OCCURRENCE AS con 

	JOIN (
		SELECT vit.PERSON_ID, coh.outcome, vit.VISIT_OCCURRENCE_ID, vit.VISIT_START_DATE, ROW_NUMBER() OVER (PARTITION BY vit.person_id ORDER BY vit.visit_start_date asc) AS visit_seq
			FROM @cdmDatabaseSchema.VISIT_OCCURRENCE vit
			JOIN #cohort coh
			ON vit.person_id = coh.subject_id
				WHERE coh.cohort_definition_id = @cohort_definition_id
				AND vit.visit_start_date >= coh.cohort_start_date
				AND vit.visit_end_date <= coh.cohort_end_date
		) AS X
	ON con.visit_occurrence_id = X.visit_occurrence_id

	JOIN @cdmDatabaseSchema.PERSON AS per
	ON con.person_id = per.person_id

	--LEFT JOIN @cdmDatabaseSchema.concept_relationship AS rel
	--ON con.condition_concept_id = rel.concept_id_1
	--WHERE rel.RELATIONSHIP_ID = 'Is a'
	ORDER BY person_id, visit_seq, dx_seq
;