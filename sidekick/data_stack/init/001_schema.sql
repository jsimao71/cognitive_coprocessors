CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE products(
  product text PRIMARY KEY,
  category text NOT NULL,
  owner_id integer NOT NULL
);
CREATE TABLE owners(owner_id integer PRIMARY KEY, owner text NOT NULL);
CREATE TABLE sales(product text NOT NULL, year integer NOT NULL, amount integer NOT NULL);
CREATE TABLE reports(
  identifier text PRIMARY KEY,
  collection text NOT NULL,
  content text NOT NULL,
  embedding vector(3) NOT NULL
);

INSERT INTO products VALUES
  ('Aster', 'hardware', 1),
  ('Birch', 'software', 2),
  ('Cedar', 'hardware', 3);
INSERT INTO owners VALUES (1, 'Mina'), (2, 'Rui'), (3, 'Tala');
INSERT INTO sales VALUES
  ('Aster', 2025, 80), ('Aster', 2026, 125),
  ('Birch', 2025, 110), ('Birch', 2026, 205),
  ('Cedar', 2025, 95), ('Cedar', 2026, 170);
INSERT INTO reports VALUES
  ('report_margin_q2', 'reports',
   'Second-quarter margin fell because freight expenses rose and discounting increased.',
   '[1,0,0]'),
  ('report_churn_q3', 'reports',
   'Customer attrition improved after onboarding was simplified and response time fell.',
   '[0,1,0]'),
  ('report_energy_q1', 'reports',
   'Electricity consumption declined after cooling controls and server consolidation.',
   '[0,0,1]');

CREATE INDEX reports_collection_idx ON reports(collection);
