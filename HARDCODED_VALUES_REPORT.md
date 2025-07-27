🔍 HARDCODED VALUES AUDIT REPORT
============================================================
Total hardcoded values found: 281

📁 HOST VALUES (34 found)
----------------------------------------
📄 /workspaces/universal-knowledge-hub/manage_api.py
   Line 22: 0.0.0.0
   → Replace with environment variable for bind address

   Line 24: localhost
   → Replace with environment variable for local hostname

📄 /workspaces/universal-knowledge-hub/start_api.py
   Line 27: 0.0.0.0
   → Replace with environment variable for bind address

📄 /workspaces/universal-knowledge-hub/agents/lead_orchestrator.py
   Line 1460: 0.0.0.0
   → Replace with environment variable for bind address

📄 /workspaces/universal-knowledge-hub/agents/retrieval_agent.py
   Line 213: localhost
   → Replace with environment variable for local hostname

   Line 350: localhost
   → Replace with environment variable for local hostname

   Line 355: localhost
   → Replace with environment variable for local hostname

   Line 360: localhost
   → Replace with environment variable for local hostname

   Line 831: localhost
   → Replace with environment variable for local hostname

   Line 836: localhost
   → Replace with environment variable for local hostname

   Line 842: localhost
   → Replace with environment variable for local hostname

📄 /workspaces/universal-knowledge-hub/api/main.py
   Line 456: 0.0.0.0
   → Replace with environment variable for bind address

📄 /workspaces/universal-knowledge-hub/api/recommendation_service.py
   Line 314: localhost
   → Replace with environment variable for local hostname

📄 /workspaces/universal-knowledge-hub/scripts/check_hardcoded_values.py
   Line 26: localhost
   → Replace with environment variable for local hostname

   Line 27: localhost
   → Replace with environment variable for local hostname

   Line 28: localhost
   → Replace with environment variable for local hostname

   Line 29: localhost
   → Replace with environment variable for local hostname

   Line 30: localhost
   → Replace with environment variable for local hostname

   Line 73: localhost
   → Replace with environment variable for local hostname

📄 /workspaces/universal-knowledge-hub/tests/run_simple_tests.py
   Line 136: localhost
   → Replace with environment variable for local hostname

   Line 189: localhost
   → Replace with environment variable for local hostname

📄 /workspaces/universal-knowledge-hub/tests/test_bulletproof_comprehensive.py
   Line 57: localhost
   → Replace with environment variable for local hostname

📄 /workspaces/universal-knowledge-hub/tests/test_complete_system.py
   Line 337: localhost
   → Replace with environment variable for local hostname

   Line 636: 127.0.0.1
   → Replace with environment variable for local ip

   Line 659: 127.0.0.1
   → Replace with environment variable for local ip

   Line 695: 127.0.0.1
   → Replace with environment variable for local ip

📄 /workspaces/universal-knowledge-hub/tests/test_comprehensive.py
   Line 352: 127.0.0.1
   → Replace with environment variable for local ip

   Line 378: 127.0.0.1
   → Replace with environment variable for local ip

   Line 523: localhost
   → Replace with environment variable for local hostname

📄 /workspaces/universal-knowledge-hub/tests/test_configuration.py
   Line 222: localhost
   → Replace with environment variable for local hostname

📄 /workspaces/universal-knowledge-hub/tests/test_load_stress_performance.py
   Line 43: localhost
   → Replace with environment variable for local hostname

📄 /workspaces/universal-knowledge-hub/tests/test_security.py
   Line 127: 127.0.0.1
   → Replace with environment variable for local ip

📄 /workspaces/universal-knowledge-hub/tests/test_security_comprehensive.py
   Line 39: localhost
   → Replace with environment variable for local hostname

📄 /workspaces/universal-knowledge-hub/tests/test_simple_bulletproof.py
   Line 32: localhost
   → Replace with environment variable for local hostname

📁 INTERVAL VALUES (3 found)
----------------------------------------
📄 /workspaces/universal-knowledge-hub/manage_api.py
   Line 57: sleep(3)
   → Replace with environment variable for sleep interval

   Line 118: sleep(2)
   → Replace with environment variable for short sleep

📄 /workspaces/universal-knowledge-hub/tests/test_bulletproof_comprehensive.py
   Line 607: sleep(2)
   → Replace with environment variable for short sleep

📁 TIMEOUT VALUES (17 found)
----------------------------------------
📄 /workspaces/universal-knowledge-hub/manage_api.py
   Line 86: timeout=5
   → Replace with environment variable for quick timeout

📄 /workspaces/universal-knowledge-hub/api/main.py
   Line 328: timeout=30
   → Replace with environment variable for request timeout

📄 /workspaces/universal-knowledge-hub/scripts/check_hardcoded_values.py
   Line 45: timeout=30
   → Replace with environment variable for request timeout

   Line 46: timeout=10
   → Replace with environment variable for short timeout

   Line 47: timeout=5
   → Replace with environment variable for quick timeout

📄 /workspaces/universal-knowledge-hub/tests/run_simple_tests.py
   Line 141: timeout=10
   → Replace with environment variable for short timeout

   Line 156: timeout=10
   → Replace with environment variable for short timeout

   Line 170: timeout=10
   → Replace with environment variable for short timeout

   Line 189: timeout=10
   → Replace with environment variable for short timeout

📄 /workspaces/universal-knowledge-hub/tests/test_complete_system.py
   Line 355: timeout=30
   → Replace with environment variable for request timeout

📄 /workspaces/universal-knowledge-hub/tests/test_comprehensive.py
   Line 540: timeout=30
   → Replace with environment variable for request timeout

📄 /workspaces/universal-knowledge-hub/tests/test_integration.py
   Line 535: timeout=30
   → Replace with environment variable for request timeout

📄 /workspaces/universal-knowledge-hub/tests/test_load_stress_performance.py
   Line 84: timeout=30
   → Replace with environment variable for request timeout

   Line 86: timeout=30
   → Replace with environment variable for request timeout

   Line 277: timeout=10
   → Replace with environment variable for short timeout

   Line 331: timeout=10
   → Replace with environment variable for short timeout

   Line 477: timeout=10
   → Replace with environment variable for short timeout

📁 BUDGET VALUES (120 found)
----------------------------------------
📄 /workspaces/universal-knowledge-hub/agents/base_agent.py
   Line 88: 1000
   → Replace with environment variable for default token budget

📄 /workspaces/universal-knowledge-hub/agents/citation_agent.py
   Line 39: 1000
   → Replace with environment variable for default token budget

   Line 48: 1000
   → Replace with environment variable for default token budget

📄 /workspaces/universal-knowledge-hub/agents/factcheck_agent.py
   Line 39: 1000
   → Replace with environment variable for default token budget

   Line 47: 1000
   → Replace with environment variable for default token budget

📄 /workspaces/universal-knowledge-hub/agents/lead_orchestrator.py
   Line 251: 1000
   → Replace with environment variable for default token budget

   Line 267: 1000
   → Replace with environment variable for default token budget

   Line 392: 1000
   → Replace with environment variable for default token budget

   Line 401: 1000
   → Replace with environment variable for default token budget

   Line 490: 1000
   → Replace with environment variable for default token budget

   Line 499: 1000
   → Replace with environment variable for default token budget

   Line 566: 1000
   → Replace with environment variable for default token budget

   Line 575: 1000
   → Replace with environment variable for default token budget

   Line 699: 1000
   → Replace with environment variable for default token budget

   Line 708: 1000
   → Replace with environment variable for default token budget

   Line 738: 1000
   → Replace with environment variable for default token budget

   Line 858: 1000
   → Replace with environment variable for default token budget

   Line 1118: 1000000
   → Replace with environment variable for daily token budget

   Line 1118: 10000
   → Replace with environment variable for max tokens per query

   Line 1118: 1000
   → Replace with environment variable for default token budget

   Line 1131: 1000
   → Replace with environment variable for default token budget

   Line 1168: 10000
   → Replace with environment variable for max tokens per query

   Line 1168: 1000
   → Replace with environment variable for default token budget

📄 /workspaces/universal-knowledge-hub/agents/retrieval_agent.py
   Line 391: 1000
   → Replace with environment variable for default token budget

   Line 405: 1000
   → Replace with environment variable for default token budget

   Line 431: 1000
   → Replace with environment variable for default token budget

   Line 445: 1000
   → Replace with environment variable for default token budget

   Line 480: 1000
   → Replace with environment variable for default token budget

   Line 494: 1000
   → Replace with environment variable for default token budget

   Line 600: 1000
   → Replace with environment variable for default token budget

📄 /workspaces/universal-knowledge-hub/agents/synthesis_agent.py
   Line 38: 1000
   → Replace with environment variable for default token budget

   Line 47: 1000
   → Replace with environment variable for default token budget

📄 /workspaces/universal-knowledge-hub/api/analytics.py
   Line 55: 10000
   → Replace with environment variable for max tokens per query

   Line 55: 1000
   → Replace with environment variable for default token budget

   Line 260: 1000
   → Replace with environment variable for default token budget

📄 /workspaces/universal-knowledge-hub/api/cache.py
   Line 53: 1000
   → Replace with environment variable for default token budget

📄 /workspaces/universal-knowledge-hub/api/main.py
   Line 169: 10000
   → Replace with environment variable for max tokens per query

   Line 169: 1000
   → Replace with environment variable for default token budget

   Line 171: 10000
   → Replace with environment variable for max tokens per query

   Line 171: 1000
   → Replace with environment variable for default token budget

   Line 171: 1000
   → Replace with environment variable for default token budget

   Line 182: 10000
   → Replace with environment variable for max tokens per query

   Line 182: 1000
   → Replace with environment variable for default token budget

   Line 267: 10000
   → Replace with environment variable for max tokens per query

   Line 267: 1000
   → Replace with environment variable for default token budget

   Line 273: 10000
   → Replace with environment variable for max tokens per query

   Line 273: 1000
   → Replace with environment variable for default token budget

   Line 315: 1000
   → Replace with environment variable for default token budget

   Line 318: 1000
   → Replace with environment variable for default token budget

📄 /workspaces/universal-knowledge-hub/api/security.py
   Line 237: 1000
   → Replace with environment variable for default token budget

   Line 258: 1000
   → Replace with environment variable for default token budget

📄 /workspaces/universal-knowledge-hub/api/middleware/security.py
   Line 29: 1000
   → Replace with environment variable for default token budget

   Line 30: 10000
   → Replace with environment variable for max tokens per query

   Line 30: 1000
   → Replace with environment variable for default token budget

   Line 236: 1000
   → Replace with environment variable for default token budget

   Line 237: 10000
   → Replace with environment variable for max tokens per query

   Line 237: 1000
   → Replace with environment variable for default token budget

📄 /workspaces/universal-knowledge-hub/scripts/check_hardcoded_values.py
   Line 56: 1000
   → Replace with environment variable for default token budget

   Line 57: 10000
   → Replace with environment variable for max tokens per query

   Line 57: 1000
   → Replace with environment variable for default token budget

   Line 63: 1000000
   → Replace with environment variable for daily token budget

   Line 63: 10000
   → Replace with environment variable for max tokens per query

   Line 63: 1000
   → Replace with environment variable for default token budget

   Line 64: 10000
   → Replace with environment variable for max tokens per query

   Line 64: 1000
   → Replace with environment variable for default token budget

   Line 65: 1000
   → Replace with environment variable for default token budget

📄 /workspaces/universal-knowledge-hub/tests/run_simple_tests.py
   Line 192: 1000
   → Replace with environment variable for default token budget

   Line 195: 1000
   → Replace with environment variable for default token budget

📄 /workspaces/universal-knowledge-hub/tests/test_bulletproof_comprehensive.py
   Line 514: 1000
   → Replace with environment variable for default token budget

   Line 539: 1000
   → Replace with environment variable for default token budget

   Line 615: 1000
   → Replace with environment variable for default token budget

📄 /workspaces/universal-knowledge-hub/tests/test_complete_system.py
   Line 107: 10000
   → Replace with environment variable for max tokens per query

   Line 107: 1000
   → Replace with environment variable for default token budget

   Line 117: 10000
   → Replace with environment variable for max tokens per query

   Line 117: 1000
   → Replace with environment variable for default token budget

   Line 277: 1000
   → Replace with environment variable for default token budget

   Line 870: 1000
   → Replace with environment variable for default token budget

   Line 876: 1000
   → Replace with environment variable for default token budget

   Line 960: 1000
   → Replace with environment variable for default token budget

   Line 965: 1000
   → Replace with environment variable for default token budget

📄 /workspaces/universal-knowledge-hub/tests/test_comprehensive.py
   Line 90: 10000
   → Replace with environment variable for max tokens per query

   Line 90: 1000
   → Replace with environment variable for default token budget

   Line 100: 10000
   → Replace with environment variable for max tokens per query

   Line 100: 1000
   → Replace with environment variable for default token budget

   Line 465: 1000
   → Replace with environment variable for default token budget

📄 /workspaces/universal-knowledge-hub/tests/test_configuration.py
   Line 418: 1000
   → Replace with environment variable for default token budget

   Line 425: 1000
   → Replace with environment variable for default token budget

📄 /workspaces/universal-knowledge-hub/tests/test_integration.py
   Line 74: 1000
   → Replace with environment variable for default token budget

   Line 478: 10000
   → Replace with environment variable for max tokens per query

   Line 478: 1000
   → Replace with environment variable for default token budget

📄 /workspaces/universal-knowledge-hub/tests/test_lead_orchestrator.py
   Line 30: 1000
   → Replace with environment variable for default token budget

📄 /workspaces/universal-knowledge-hub/tests/test_load.py
   Line 130: 1000
   → Replace with environment variable for default token budget

   Line 261: 1000
   → Replace with environment variable for default token budget

📄 /workspaces/universal-knowledge-hub/tests/test_load_stress_performance.py
   Line 47: 1000
   → Replace with environment variable for default token budget

   Line 89: 1000
   → Replace with environment variable for default token budget

   Line 104: 1000
   → Replace with environment variable for default token budget

   Line 315: 1000
   → Replace with environment variable for default token budget

   Line 360: 1000
   → Replace with environment variable for default token budget

   Line 361: 1000
   → Replace with environment variable for default token budget

   Line 382: 1000
   → Replace with environment variable for default token budget

   Line 383: 1000
   → Replace with environment variable for default token budget

📄 /workspaces/universal-knowledge-hub/tests/test_performance.py
   Line 351: 1000
   → Replace with environment variable for default token budget

   Line 361: 1000
   → Replace with environment variable for default token budget

   Line 371: 1000
   → Replace with environment variable for default token budget

   Line 456: 10000
   → Replace with environment variable for max tokens per query

   Line 456: 1000
   → Replace with environment variable for default token budget

📄 /workspaces/universal-knowledge-hub/tests/test_prompts_comprehensive.py
   Line 379: 1000
   → Replace with environment variable for default token budget

   Line 381: 1000
   → Replace with environment variable for default token budget

   Line 451: 1000
   → Replace with environment variable for default token budget

📄 /workspaces/universal-knowledge-hub/tests/test_security.py
   Line 112: 1000000
   → Replace with environment variable for daily token budget

   Line 112: 10000
   → Replace with environment variable for max tokens per query

   Line 112: 1000
   → Replace with environment variable for default token budget

   Line 299: 1000
   → Replace with environment variable for default token budget

📄 /workspaces/universal-knowledge-hub/tests/test_security_comprehensive.py
   Line 402: 1000
   → Replace with environment variable for default token budget

📄 /workspaces/universal-knowledge-hub/tests/test_simple_bulletproof.py
   Line 36: 1000
   → Replace with environment variable for default token budget

   Line 267: 1000
   → Replace with environment variable for default token budget

📄 /workspaces/universal-knowledge-hub/tests/performance/locustfile.py
   Line 112: 1000
   → Replace with environment variable for default token budget

   Line 163: 1000
   → Replace with environment variable for default token budget

   Line 229: 1000
   → Replace with environment variable for default token budget

   Line 365: 1000
   → Replace with environment variable for default token budget

📁 DB VALUES (14 found)
----------------------------------------
📄 /workspaces/universal-knowledge-hub/agents/factcheck_agent.py
   Line 16: knowledge_base
   → Replace with environment variable for database name

📄 /workspaces/universal-knowledge-hub/agents/lead_orchestrator.py
   Line 374: knowledge_base
   → Replace with environment variable for database name

📄 /workspaces/universal-knowledge-hub/agents/retrieval_agent.py
   Line 155: knowledge_base
   → Replace with environment variable for database name

   Line 352: knowledge_base
   → Replace with environment variable for database name

   Line 357: knowledge_base
   → Replace with environment variable for database name

   Line 360: knowledge_base
   → Replace with environment variable for database name

   Line 833: knowledge_base
   → Replace with environment variable for database name

   Line 839: knowledge_base
   → Replace with environment variable for database name

📄 /workspaces/universal-knowledge-hub/scripts/check_hardcoded_values.py
   Line 78: knowledge_base
   → Replace with environment variable for database name

   Line 79: ukp_db
   → Replace with environment variable for application database

   Line 80: universal-knowledge-hub
   → Replace with environment variable for index name

📄 /workspaces/universal-knowledge-hub/tests/test_agents.py
   Line 269: knowledge_base
   → Replace with environment variable for database name

📄 /workspaces/universal-knowledge-hub/tests/performance/locustfile.py
   Line 410: universal-knowledge-hub
   → Replace with environment variable for index name

   Line 419: universal-knowledge-hub
   → Replace with environment variable for index name

📁 THRESHOLD VALUES (33 found)
----------------------------------------
📄 /workspaces/universal-knowledge-hub/agents/factcheck_agent.py
   Line 104: 0.7
   → Replace with environment variable for confidence threshold

📄 /workspaces/universal-knowledge-hub/agents/lead_orchestrator.py
   Line 285: 0.95
   → Replace with environment variable for high similarity threshold

   Line 564: 0.95
   → Replace with environment variable for high similarity threshold

   Line 1032: 0.7
   → Replace with environment variable for confidence threshold

   Line 1167: 0.95
   → Replace with environment variable for high similarity threshold

📄 /workspaces/universal-knowledge-hub/agents/retrieval_agent.py
   Line 107: 0.95
   → Replace with environment variable for high similarity threshold

   Line 243: 0.92
   → Replace with environment variable for similarity threshold

   Line 330: 0.92
   → Replace with environment variable for similarity threshold

   Line 348: 0.92
   → Replace with environment variable for similarity threshold

📄 /workspaces/universal-knowledge-hub/api/main.py
   Line 172: 0.7
   → Replace with environment variable for confidence threshold

   Line 317: 0.7
   → Replace with environment variable for confidence threshold

📄 /workspaces/universal-knowledge-hub/tests/test_agents.py
   Line 218: 0.7
   → Replace with environment variable for confidence threshold

   Line 322: 0.7
   → Replace with environment variable for confidence threshold

   Line 393: 0.7
   → Replace with environment variable for confidence threshold

📄 /workspaces/universal-knowledge-hub/tests/test_bulletproof_comprehensive.py
   Line 353: 0.7
   → Replace with environment variable for confidence threshold

📄 /workspaces/universal-knowledge-hub/tests/test_complete_system.py
   Line 146: 0.95
   → Replace with environment variable for high similarity threshold

   Line 154: 0.95
   → Replace with environment variable for high similarity threshold

   Line 278: 0.7
   → Replace with environment variable for confidence threshold

   Line 425: 0.95
   → Replace with environment variable for high similarity threshold

   Line 433: 0.95
   → Replace with environment variable for high similarity threshold

📄 /workspaces/universal-knowledge-hub/tests/test_comprehensive.py
   Line 132: 0.95
   → Replace with environment variable for high similarity threshold

   Line 139: 0.95
   → Replace with environment variable for high similarity threshold

   Line 466: 0.7
   → Replace with environment variable for confidence threshold

📄 /workspaces/universal-knowledge-hub/tests/test_core_functionality.py
   Line 83: 0.95
   → Replace with environment variable for high similarity threshold

   Line 88: 0.95
   → Replace with environment variable for high similarity threshold

📄 /workspaces/universal-knowledge-hub/tests/test_integration.py
   Line 335: 0.95
   → Replace with environment variable for high similarity threshold

📄 /workspaces/universal-knowledge-hub/tests/test_load.py
   Line 117: 0.7
   → Replace with environment variable for confidence threshold

📄 /workspaces/universal-knowledge-hub/tests/test_load_stress_performance.py
   Line 409: 0.95
   → Replace with environment variable for high similarity threshold

📄 /workspaces/universal-knowledge-hub/tests/test_prompts_comprehensive.py
   Line 176: 0.7
   → Replace with environment variable for confidence threshold

   Line 216: 0.7
   → Replace with environment variable for confidence threshold

📄 /workspaces/universal-knowledge-hub/tests/test_recommendation_system.py
   Line 322: 0.7
   → Replace with environment variable for confidence threshold

   Line 324: 0.7
   → Replace with environment variable for confidence threshold

📄 /workspaces/universal-knowledge-hub/tests/performance/locustfile.py
   Line 83: 0.7
   → Replace with environment variable for confidence threshold

📁 URL VALUES (13 found)
----------------------------------------
📄 /workspaces/universal-knowledge-hub/agents/retrieval_agent.py
   Line 213: http://localhost:8890
   → Replace with environment variable for database/service url

   Line 360: http://localhost:7200
   → Replace with environment variable for database/service url

   Line 831: http://localhost:6333
   → Replace with environment variable for database/service url

   Line 836: http://localhost:9200
   → Replace with environment variable for database/service url

   Line 842: http://localhost:8890
   → Replace with environment variable for database/service url

📄 /workspaces/universal-knowledge-hub/api/recommendation_service.py
   Line 314: bolt://localhost:7687
   → Replace with environment variable for neo4j connection url

📄 /workspaces/universal-knowledge-hub/tests/run_simple_tests.py
   Line 136: http://localhost:8003
   → Replace with environment variable for database/service url

   Line 189: http://localhost:8003
   → Replace with environment variable for database/service url

📄 /workspaces/universal-knowledge-hub/tests/test_bulletproof_comprehensive.py
   Line 57: http://localhost:8003
   → Replace with environment variable for database/service url

📄 /workspaces/universal-knowledge-hub/tests/test_complete_system.py
   Line 337: bolt://localhost:7687
   → Replace with environment variable for neo4j connection url

📄 /workspaces/universal-knowledge-hub/tests/test_comprehensive.py
   Line 523: bolt://localhost:7687
   → Replace with environment variable for neo4j connection url

📄 /workspaces/universal-knowledge-hub/tests/test_load_stress_performance.py
   Line 43: http://localhost:8003
   → Replace with environment variable for database/service url

📄 /workspaces/universal-knowledge-hub/tests/test_security_comprehensive.py
   Line 39: http://localhost:8003
   → Replace with environment variable for database/service url

📁 PORT VALUES (23 found)
----------------------------------------
📄 /workspaces/universal-knowledge-hub/agents/retrieval_agent.py
   Line 213: :8890
   → Replace with environment variable for sparql endpoint port

   Line 360: :7200
   → Replace with environment variable for knowledge graph port

   Line 831: :6333
   → Replace with environment variable for vector db port

   Line 836: :9200
   → Replace with environment variable for elasticsearch port

   Line 842: :8890
   → Replace with environment variable for sparql endpoint port

📄 /workspaces/universal-knowledge-hub/api/recommendation_service.py
   Line 314: :7687
   → Replace with environment variable for neo4j port

📄 /workspaces/universal-knowledge-hub/scripts/check_hardcoded_values.py
   Line 33: :8002
   → Replace with environment variable for api port

   Line 34: :8003
   → Replace with environment variable for test api port

   Line 35: :8000
   → Replace with environment variable for development port

   Line 36: :6333
   → Replace with environment variable for vector db port

   Line 37: :9200
   → Replace with environment variable for elasticsearch port

   Line 38: :7687
   → Replace with environment variable for neo4j port

   Line 39: :8890
   → Replace with environment variable for sparql endpoint port

   Line 40: :7200
   → Replace with environment variable for knowledge graph port

   Line 41: :6379
   → Replace with environment variable for redis port

   Line 42: :5432
   → Replace with environment variable for postgresql port

📄 /workspaces/universal-knowledge-hub/tests/run_simple_tests.py
   Line 136: :8003
   → Replace with environment variable for test api port

   Line 189: :8003
   → Replace with environment variable for test api port

📄 /workspaces/universal-knowledge-hub/tests/test_bulletproof_comprehensive.py
   Line 57: :8003
   → Replace with environment variable for test api port

📄 /workspaces/universal-knowledge-hub/tests/test_complete_system.py
   Line 337: :7687
   → Replace with environment variable for neo4j port

📄 /workspaces/universal-knowledge-hub/tests/test_comprehensive.py
   Line 523: :7687
   → Replace with environment variable for neo4j port

📄 /workspaces/universal-knowledge-hub/tests/test_load_stress_performance.py
   Line 43: :8003
   → Replace with environment variable for test api port

📄 /workspaces/universal-knowledge-hub/tests/test_security_comprehensive.py
   Line 39: :8003
   → Replace with environment variable for test api port

📁 TTL VALUES (8 found)
----------------------------------------
📄 /workspaces/universal-knowledge-hub/scripts/check_hardcoded_values.py
   Line 50: ttl=3600
   → Replace with environment variable for cache ttl

   Line 51: ttl=7200
   → Replace with environment variable for long cache ttl

   Line 52: ttl=1800
   → Replace with environment variable for short cache ttl

📄 /workspaces/universal-knowledge-hub/tests/test_complete_system.py
   Line 722: ttl=3600
   → Replace with environment variable for cache ttl

   Line 726: ttl=3600
   → Replace with environment variable for cache ttl

   Line 727: ttl=3600
   → Replace with environment variable for cache ttl

   Line 728: ttl=3600
   → Replace with environment variable for cache ttl

   Line 872: ttl=3600
   → Replace with environment variable for cache ttl

📁 LIMIT VALUES (16 found)
----------------------------------------
📄 /workspaces/universal-knowledge-hub/scripts/check_hardcoded_values.py
   Line 55: max_workers=5
   → Replace with environment variable for concurrent workers

   Line 56: max_size=1000
   → Replace with environment variable for cache size

   Line 57: max_size=1000
   → Replace with environment variable for cache size

   Line 57: max_size=10000
   → Replace with environment variable for large cache size

   Line 58: limit=10
   → Replace with environment variable for result limit

   Line 59: limit=20
   → Replace with environment variable for large result limit

   Line 60: limit=50
   → Replace with environment variable for very large result limit

📄 /workspaces/universal-knowledge-hub/tests/test_bulletproof_comprehensive.py
   Line 258: limit=10
   → Replace with environment variable for result limit

   Line 270: limit=10
   → Replace with environment variable for result limit

   Line 282: limit=10
   → Replace with environment variable for result limit

   Line 761: limit=10
   → Replace with environment variable for result limit

📄 /workspaces/universal-knowledge-hub/tests/test_complete_system.py
   Line 870: max_size=1000
   → Replace with environment variable for cache size

📄 /workspaces/universal-knowledge-hub/tests/test_load.py
   Line 246: max_workers=5
   → Replace with environment variable for concurrent workers

📄 /workspaces/universal-knowledge-hub/tests/test_recommendation_system.py
   Line 350: limit=10
   → Replace with environment variable for result limit

   Line 396: limit=10
   → Replace with environment variable for result limit

   Line 405: limit=10
   → Replace with environment variable for result limit

🔧 SUGGESTED ENVIRONMENT VARIABLES
============================================================
   API_PORT_PORT
   APPLICATION_DATABASE_NAME
   BIND_ADDRESS_HOST
   CACHE_TTL_TTL
   CONFIDENCE_THRESHOLD_THRESHOLD
   DAILY_TOKEN_BUDGET
   DATABASE/SERVICE_URL_URL
   DATABASE_NAME_NAME
   DEFAULT_TOKEN_BUDGET
   DEVELOPMENT_PORT_PORT
   ELASTICSEARCH_PORT_PORT
   HIGH_SIMILARITY_THRESHOLD_THRESHOLD
   INDEX_NAME_NAME
   KNOWLEDGE_GRAPH_PORT_PORT
   LOCAL_HOSTNAME_HOST
   LOCAL_IP_HOST
   LONG_CACHE_TTL_TTL
   MAX_CACHE_SIZE
   MAX_CONCURRENT_WORKERS
   MAX_LARGE_CACHE_SIZE
   MAX_LARGE_RESULT_LIMIT
   MAX_RESULT_LIMIT
   MAX_TOKENS_PER_QUERY
   MAX_VERY_LARGE_RESULT_LIMIT
   NEO4J_CONNECTION_URL_URL
   NEO4J_PORT_PORT
   POSTGRESQL_PORT_PORT
   QUICK_TIMEOUT_TIMEOUT
   REDIS_PORT_PORT
   REQUEST_TIMEOUT_TIMEOUT
   SHORT_CACHE_TTL_TTL
   SHORT_TIMEOUT_TIMEOUT
   SIMILARITY_THRESHOLD_THRESHOLD
   SPARQL_ENDPOINT_PORT_PORT
   TEST_API_PORT_PORT
   VECTOR_DB_PORT_PORT