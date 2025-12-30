curl -s http://0.0.0.0:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Ministral-3-14B-Instruct-2512",
    "prompt": "Translate from Northern Sami to Finnish\n\nNorthern Sami sentence: Čoakkán mearridii ovddidit oassebeallečoakkámii, ahte čuovvovaš artihkal 8 (j) - čoakkámis livččii čiekŋalis dialoga Abs-áššiin ja juogo suodjaluvvon guovlluin dehe biodiversiteahtas ja dálkkádatrievdamis.\n\nFinnish sentence: ",
    "max_tokens": 100,
    "temperature": 0.7
  }' | jq
