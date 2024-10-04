-- drop any previous view if it was existing
DROP VIEW IF EXISTS gmdn_emdn_id_best_correspondence;
DROP VIEW IF EXISTS gmdn_root_emdn_id_best_correspondence;
DROP VIEW IF EXISTS gmdn_root_emdn_correspondence;
DROP VIEW IF EXISTS emdn_id_root_emdn_id;



-- create a view to get the correspondence between each EMDN code and the
-- root EMDN code
CREATE VIEW emdn_id_root_emdn_id AS
SELECT e2.identifier AS emdn_id,
       e2.code       AS emdn_code_string,
       e1.identifier AS root_id,
       emdn_category AS root_category
FROM emdn_code as e1,
     (SELECT *, SUBSTRING(e_code.code, 1, 1) as emdn_category
      FROM emdn_code as e_code) AS e2
WHERE e1.code = e2.emdn_category;

-- create a view that keeps the correspondence between GMDN code and
-- EMDN root codes, with the related frequency
CREATE VIEW gmdn_root_emdn_correspondence AS
SELECT g_e_cat.gmdn_id       AS gmdn_id,
       emdn_category         AS emdn_category,
       e_code_ext.identifier AS root_emdn_id,
       description           AS emdn_description,
       COUNT(*)              as count
FROM (SELECT mapping.gmdn_id,
             SUBSTRING(e_code.code, 1, 1) as emdn_category
      FROM mapping as mapping,
           emdn_code as e_code
      WHERE mapping.emdn_id = e_code.identifier) as g_e_cat,
     emdn_code as e_code_ext
WHERE emdn_category = e_code_ext.code
GROUP BY g_e_cat.gmdn_id, g_e_cat.emdn_category, e_code_ext.identifier, description;

-- create a view that contains, for each GMDN code, the best 2 correspondences
-- with EMDN root codes
CREATE VIEW gmdn_root_emdn_id_best_correspondence AS
WITH ranked_frequencies AS (SELECT *,
                                   ROW_NUMBER() OVER (
                                       PARTITION BY gmdn_id ORDER BY count DESC
                                       ) AS rn
                            FROM gmdn_root_emdn_correspondence)
SELECT gmdn_id,
       root_emdn_id,
       count
FROM ranked_frequencies
WHERE rn <= 2
ORDER BY gmdn_id,
         rn DESC;

-- verify that each GMDN code has at most 2 corresponding EMDN codes
SELECT gmdn_id, COUNT(*)
FROM gmdn_root_emdn_id_best_correspondence
GROUP BY gmdn_id
HAVING COUNT(*) > 2;

-- join the mapping table with the gmdn_root_emdn_id_best_correspondence view
-- and the emdn_id_root_emdn_id view to get the final mappings;
-- for each GMDN code, two mappings are kept (one for each EMDN root code),
-- that correspond to the longest EMDN code (most specific ones
CREATE VIEW gmdn_emdn_id_best_correspondence AS
WITH ranked_correspondences AS (SELECT m.identifier                     AS mapping_id,
                                       best.gmdn_id                     AS gmdn_id,
                                       emdn_id_root_id.emdn_id          AS emdn_id,
                                       emdn_id_root_id.emdn_code_string AS emdn_code_string,
                                       emdn_id_root_id.root_category    AS root_category,
                                       ROW_NUMBER() OVER (
                                           PARTITION BY best.gmdn_id, best.root_emdn_id
                                           ORDER BY LENGTH(emdn_id_root_id.emdn_code_string) DESC
                                           )                            AS rn
                                FROM gmdn_root_emdn_id_best_correspondence AS best
                                         JOIN mapping AS m
                                              ON m.gmdn_id = best.gmdn_id
                                         JOIN emdn_id_root_emdn_id AS emdn_id_root_id
                                              on best.root_emdn_id = emdn_id_root_id.root_id
                                WHERE m.emdn_id = emdn_id_root_id.emdn_id)
SELECT mapping_id,
       gmdn_id,
       emdn_id,
       emdn_code_string,
       root_category
FROM ranked_correspondences
WHERE rn < 2;

-- group by the gmdn_id and sum the frequencies
SELECT gmdn_id, COUNT(*)
FROM gmdn_emdn_id_best_correspondence
GROUP BY gmdn_id

