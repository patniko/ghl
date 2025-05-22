# Progress Hub

## Run Locally

Clone the project

```bash
  git clone https://github.com/satnaing/shadcn-admin.git
```

Go to the project directory

```bash
  cd shadcn-admin
```

Install dependencies

```bash
  pnpm install
```

Start the server

```bash
  pnpm run dev
```

## Running with API and Tasks

For a complete development environment, you can run the UI together with the API and task services using the combined commands from the ph-api directory:

```bash
# From the ph-api directory
make run-all            # Runs UI, API, and Tasks as separate processes
make run-all-integrated # Runs UI and API with integrated tasks
```

These commands will start all necessary services together, making it easier to develop and test the full application stack.

## Tech Stack

**UI:** [ShadcnUI](https://ui.shadcn.com) (TailwindCSS + RadixUI)

**Build Tool:** [Vite](https://vitejs.dev/)

**Routing:** [TanStack Router](https://tanstack.com/router/latest)

**Type Checking:** [TypeScript](https://www.typescriptlang.org/)

**Linting/Formatting:** [Eslint](https://eslint.org/) & [Prettier](https://prettier.io/)

**Icons:** [Tabler Icons](https://tabler.io/icons)
