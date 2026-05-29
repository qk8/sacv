# AGENTS.md — Living Project Blueprint

> Auto-maintained by the SACV workflow. Human edits are welcome and will be
> preserved. The workflow appends to specific sections only.

---

## Architecture Overview

_This section is updated by the workflow when new modules are added._

```
┌─────────────────────────────────────────────────────┐
│                   Presentation Layer                │
│            (Controllers / React Components)         │
└───────────────────┬─────────────────────────────────┘
                    │  calls
┌───────────────────▼─────────────────────────────────┐
│                 Application Layer                   │
│                  (Use Cases / DTOs)                 │
└───────────────────┬─────────────────────────────────┘
                    │  calls
┌───────────────────▼─────────────────────────────────┐
│                  Domain Layer                       │
│          (Entities / Value Objects / Events)        │
└───────────────────┬─────────────────────────────────┘
                    │  depends on (via interfaces)
┌───────────────────▼─────────────────────────────────┐
│              Infrastructure Layer                   │
│          (Repositories / External Services)         │
└─────────────────────────────────────────────────────┘
```

**Layer boundary rules (enforced by .dependency-cruiser.json and ArchUnit):**
- Presentation → Application → Domain → Infrastructure
- Domain layer must have ZERO framework dependencies
- Infrastructure implements Domain interfaces; never the reverse

---

## Module Conventions

### Java / Spring Boot
- Service classes: `<Name>Service` suffix, live in `application/` package
- Repository interfaces: `<Name>Repository` suffix, live in `domain/`
- Repository implementations: live in `infrastructure/`
- Use constructor injection everywhere — no field injection (`@Autowired` on fields)
- `@Transactional` belongs on Use Case layer, not service or repository

### TypeScript / React
- Components: `PascalCase.tsx`, live in `features/<domain>/components/`
- Hooks: `use<Name>.ts`, live in `features/<domain>/hooks/`
- API clients: live in `infrastructure/api/`, never imported directly by components
- Props interfaces: `<ComponentName>Props` — always explicit, never inlined

---

## Common Mistakes

_Populated automatically after each agent session. Most recent first._

<!-- SACV_LESSONS_START -->
<!-- SACV_LESSONS_END -->

---

## Architecture Decisions

_Populated automatically when new modules or patterns are introduced._

<!-- SACV_ARCH_DECISIONS_START -->
<!-- SACV_ARCH_DECISIONS_END -->

---

## Test Inventory

Tests committed by the workflow live under:

| Type | Location | Framework |
|---|---|---|
| Backend unit | `src/test/java/` | JUnit 5 + AssertJ |
| Backend API (sequence) | `tests/api/routes/` | Jest + Supertest |
| Frontend E2E | `tests/e2e/features/` | Playwright |
| Architecture | `src/test/java/…/ArchitectureTest.java` | ArchUnit |

**Test deletion is strictly prohibited.** The workflow enforces this automatically.
