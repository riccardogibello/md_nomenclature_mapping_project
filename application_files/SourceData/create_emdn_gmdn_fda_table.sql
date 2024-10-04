DROP TABLE IF EXISTS emdn_gmdn_fda;
CREATE TABLE emdn_gmdn_fda
(
    original_device_mapping_id INTEGER,
    emdn_id                    INTEGER,
    emdn_code                  VARCHAR(255),
    emdn_description           VARCHAR(255),
    emdn_category              VARCHAR(255),
    gmdn_id                    INTEGER,
    gmdn_term_name             VARCHAR(255),
    product_code               VARCHAR(255),
    device_name                VARCHAR(255),
    medical_specialty          VARCHAR(255),
    panel                      VARCHAR(255)
);
INSERT INTO emdn_gmdn_fda
SELECT dm.identifier,
       emdn_id,
       e.code,
       e.description,
       emdn_category,
       gmdn_id,
       g.term_name,
       f_c.product_code,
       f_c.device_name,
       f_c.medical_specialty,
       f_c.panel
FROM device_mapping AS dm
         JOIN (SELECT ec.clean_device_id,
                      ecd.identifier            AS emdn_id,
                      ecd.code,
                      ecd.description,
                      SUBSTRING(ecd.code, 1, 1) as emdn_category
               FROM emdn_correspondence AS ec
                        JOIN emdn_code AS ecd ON ecd.identifier = ec.emdn_id) AS e
              ON dm.first_device_id = e.clean_device_id
         JOIN (SELECT gc.clean_device_id,
                      gcd.identifier AS gmdn_id,
                      gcd.term_name
               FROM gmdn_correspondence AS gc
                        JOIN gmdn_code AS gcd ON gcd.identifier = gc.gmdn_id) AS g
              ON dm.second_device_id = g.clean_device_id
         JOIN (SELECT cd.identifier,
                      fd.product_code,
                      fd.device_name,
                      fd.medical_specialty,
                      fd.panel
               FROM clean_device AS cd
                        JOIN fda_code AS fd
                             ON cd.product_code = fd.product_code) AS f_c
              ON dm.second_device_id = f_c.identifier
GROUP BY dm.identifier,
         emdn_id,
         e.code,
         e.description,
         emdn_category,
         gmdn_id,
         g.term_name,
         f_c.product_code,
         f_c.device_name,
         f_c.medical_specialty,
         f_c.panel;
