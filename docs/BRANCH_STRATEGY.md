# Git Branch Strategy

## Branch Structure

### 🏛️ **master** - Production Branch
- **Purpose**: Stable, production-ready code
- **Access**: Only vishwas (you) can push directly
- **Updates**: Only from dev branch via pull requests

### 🔧 **dev** - Development Branch  
- **Purpose**: Integration branch for all development work
- **Access**: Only vishwas (you) can push directly
- **Updates**: Receives pull requests from feature branches
- **Source**: All feature branches branch from dev

### 👨‍💻 **vishwas_dev** - Personal Development Branch
- **Purpose**: Your personal development work
- **Access**: Only you can push
- **Updates**: Push directly, then create PR to dev
- **Workflow**: vishwas_dev → dev → master

## Workflow

### For Your Development Work:
1. Work on `vishwas_dev` branch
2. Push changes to `vishwas_dev`
3. Create PR from `vishwas_dev` → `dev`
4. Merge to `dev`
5. When ready for production: PR from `dev` → `master`

### For Others' Contributions:
1. They create feature branches from `dev`
2. They create PRs to `dev` (anyone can create PRs)
3. You review and merge to `dev`
4. You control promotion to `master`

## Current Status

✅ **master** - Established as production branch
✅ **dev** - Created as development integration branch  
✅ **vishwas_dev** - Current working branch for your changes

## Branch Protection Rules (to be set on GitHub)

### master branch:
- Require pull request reviews
- Restrict pushes to vishwas only
- Require status checks to pass

### dev branch:
- Allow pull requests from anyone
- Restrict direct pushes to vishwas only
- Require review for merging

This ensures proper code review and controlled releases while allowing collaborative development.
