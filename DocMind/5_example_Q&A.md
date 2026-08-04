# DocMind — Example Queries

Example `POST /query` requests and responses, demonstrating retrieval + generation over `input/databricks_architecture_governance.pdf`.

## 1. What is a SQL warehouse?

```bash
curl -s -X POST http://localhost:8003/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is a SQL warehouse?"}' | jq -r '.answer'
```

A SQL warehouse is purpose-built compute for SQL and BI (Business Intelligence) workloads, exposed through Databricks SQL. There are three variants of SQL warehouses—Serverless, Pro, and Classic—each providing different trade-offs in terms of control, startup latency, and feature access.

---

## 2. How does the Photon engine improve performance?

```bash
curl -s -X POST http://localhost:8003/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How photon engine improves performance?"}' | jq -r '.answer'
```

The Photon engine improves performance by providing native, vectorized query execution that replaces parts of the Spark execution engine, particularly benefiting scan-, join-, and aggregation-heavy workloads. It works transparently, meaning that existing Spark SQL and DataFrame code does not need to change to take advantage of its performance gains.

---

## 3. What does a metastore signify?

```bash
curl -s -X POST http://localhost:8003/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does a metstore signify?"}' | jq -r '.answer'
```

A metastore is the top-level container for metadata in Unity Catalog, typically one per region, attached to every workspace that should share the same governance policies.

---

## 4. How can we implement row-level security and column-level masking in Unity Catalog?

```bash
curl -s -X POST http://localhost:8003/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How can we implement Row level security and column level masking in unity catalog?"}' | jq -r '.answer'
```

Row level security and column level masking in Unity Catalog can be implemented through the use of row filters and column masks.

1. **Row Filters**: A row filter is a function that determines which rows a given caller may see. For example, you can create a function that filters rows based on the caller's region:
   ```sql
   CREATE FUNCTION prod.sales.region_filter(region STRING)
   RETURN region = current_user_region() OR is_account_group_member('global-admins');
   ALTER TABLE prod.sales.regional_orders
   SET ROW FILTER prod.sales.region_filter ON (region);
   ```

2. **Column Masks**: A column mask is a function that redacts or transforms a column's value based on the caller's identity. For instance, you can create a function that shows the full salary only to HR members, while all others see NULL:
   ```sql
   CREATE FUNCTION prod.hr.mask_salary(salary DECIMAL(10,2))
   RETURN CASE
     WHEN is_account_group_member('hr-team') THEN salary
     ELSE NULL
   END;
   ALTER TABLE prod.hr.employees
     ALTER COLUMN salary SET MASK prod.hr.mask_salary;
   ```

These functions are applied dynamically at query time and ensure that access is controlled based on user identity and group membership.

---

## 5. How to use Genie spaces?

```bash
curl -s -X POST http://localhost:8003/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How to use Genie spaces?"}' | jq -r '.answer'
```

I don't know.

*(Correct behavior — "Genie spaces" isn't covered in the source document, and the system's prompt explicitly instructs it to say so rather than hallucinate an answer.)*
