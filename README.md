┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│   User Query ──▶ OOD Classifier ─┬─▶ In-Domain ──▶ DistilGPT2 ──▶ Response    │
│                                  │                       │                      │
│                                  │                       ▼                      │
│                                  │                 NER Processing               │
│                                  │                       │                      │
│                                  │                       ▼                      │
│                                  │              Dynamic Placeholder             │
│                                  │                  Replacement                 │
│                                  │                                              │
│                                  └─▶ Out-of-Domain ──▶ Polite Rejection        │
│                                                          Response               │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
