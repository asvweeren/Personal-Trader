"""API routes for model experiment tracking."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db
from app.models.model_experiment import ModelExperiment

router = APIRouter()


@router.get("/models/experiments")
async def list_experiments(
    db: AsyncSession = Depends(get_db),
    limit: int = 50,
):
    """List all model experiments, newest first."""
    result = await db.execute(
        select(ModelExperiment)
        .order_by(ModelExperiment.created_at.desc())
        .limit(limit)
    )
    experiments = result.scalars().all()
    return {
        "experiments": [
            {
                "id": exp.id,
                "strategy_name": exp.strategy_name,
                "version": exp.version,
                "train_metrics": exp.train_metrics,
                "val_metrics": exp.val_metrics,
                "test_metrics": exp.test_metrics,
                "is_active": exp.is_active,
                "notes": exp.notes,
                "created_at": exp.created_at.isoformat() if exp.created_at else None,
            }
            for exp in experiments
        ],
        "total": len(experiments),
    }


@router.get("/models/experiments/{experiment_id}")
async def get_experiment(
    experiment_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get detailed experiment including feature importance."""
    result = await db.execute(
        select(ModelExperiment).where(ModelExperiment.id == experiment_id)
    )
    exp = result.scalar_one_or_none()
    if exp is None:
        raise HTTPException(status_code=404, detail="Experiment not found")

    return {
        "id": exp.id,
        "strategy_name": exp.strategy_name,
        "version": exp.version,
        "hyperparameters": exp.hyperparameters,
        "train_metrics": exp.train_metrics,
        "val_metrics": exp.val_metrics,
        "test_metrics": exp.test_metrics,
        "feature_importance": exp.feature_importance,
        "is_active": exp.is_active,
        "notes": exp.notes,
        "created_at": exp.created_at.isoformat() if exp.created_at else None,
    }


@router.post("/models/experiments/{experiment_id}/activate")
async def activate_experiment(
    experiment_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Activate a model experiment (hot-swap to this version)."""
    result = await db.execute(
        select(ModelExperiment).where(ModelExperiment.id == experiment_id)
    )
    exp = result.scalar_one_or_none()
    if exp is None:
        raise HTTPException(status_code=404, detail="Experiment not found")

    # Deactivate all experiments for this strategy
    await db.execute(
        update(ModelExperiment)
        .where(ModelExperiment.strategy_name == exp.strategy_name)
        .values(is_active=False)
    )

    # Activate this one
    exp.is_active = True
    await db.commit()

    return {
        "status": "activated",
        "experiment_id": exp.id,
        "version": exp.version,
        "strategy_name": exp.strategy_name,
    }
