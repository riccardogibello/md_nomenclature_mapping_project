DROP TABLE IF EXISTS gmdn_fda;
CREATE TABLE gmdn_fda
(
    gmdn_id           INTEGER,
    gmdn_term_name    VARCHAR(255),
    product_code      VARCHAR(255),
    panel             VARCHAR(255),
    medical_specialty VARCHAR(255),
    device_name       VARCHAR(255)
);

INSERT INTO gmdn_fda
SELECT gmdn_id,
       g.term_name,
       f_c.product_code,
       f_c.panel,
       f_c.medical_specialty,
       f_c.device_name
FROM gmdn_correspondence AS gc
         JOIN gmdn_code AS g ON g.identifier = gc.gmdn_id
         JOIN (SELECT cd.identifier AS clean_device_id,
                      fd.product_code,
                      fd.panel,
                      fd.medical_specialty,
                      fd.device_name
               FROM clean_device AS cd
                        JOIN fda_code AS fd
                             ON cd.product_code = fd.product_code) AS f_c
              ON gc.clean_device_id = f_c.clean_device_id
GROUP BY gmdn_id,
         g.term_name,
         f_c.product_code,
         f_c.panel,
         f_c.medical_specialty,
         f_c.device_name;