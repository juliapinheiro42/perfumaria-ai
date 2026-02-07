from sqlalchemy import String, Float, Boolean, ForeignKey, Numeric, Text, Integer
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.engine import Engine


class Base(DeclarativeBase):
    pass


class Molecule(Base):
    __tablename__ = 'molecules'

    id: Mapped[int] = mapped_column(primary_key=True)
    smiles: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    molecular_weight: Mapped[float] = mapped_column(Float, nullable=True)
    log_p: Mapped[float] = mapped_column(Float, nullable=True)

    ingredients_composition = relationship(
        "Composition", back_populates="molecule")


class Ingredient(Base):
    __tablename__ = 'ingredients'

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    category: Mapped[str] = mapped_column(String, nullable=True)
    olfactive_family: Mapped[str] = mapped_column(String, nullable=True)
    olfactive_notes: Mapped[str] = mapped_column(Text, nullable=True)
    price: Mapped[float] = mapped_column(Numeric(10, 2), nullable=True)
    ifra_limit: Mapped[float] = mapped_column(Float, nullable=True)
    is_allergen: Mapped[bool] = mapped_column(Boolean, default=False)
    complexity_tier: Mapped[int] = mapped_column(Integer, nullable=True)
    traditional_use: Mapped[str] = mapped_column(String, nullable=True)

    composition = relationship(
        "Composition", back_populates="ingredient", cascade="all, delete-orphan")
    psychophysics = relationship("Psychophysics", uselist=False,
                                 back_populates="ingredient", cascade="all, delete-orphan")
    sustainability = relationship(
        "Sustainability", uselist=False, back_populates="ingredient", cascade="all, delete-orphan")


class Composition(Base):
    __tablename__ = 'composition'

    ingredient_id: Mapped[int] = mapped_column(
        ForeignKey('ingredients.id'), primary_key=True)
    molecule_id: Mapped[int] = mapped_column(
        ForeignKey('molecules.id'), primary_key=True)
    quantity: Mapped[float] = mapped_column(Float, default=1.0)

    ingredient = relationship("Ingredient", back_populates="composition")
    molecule = relationship(
        "Molecule", back_populates="ingredients_composition")


class Psychophysics(Base):
    __tablename__ = 'psychophysics'

    ingredient_id: Mapped[int] = mapped_column(
        ForeignKey('ingredients.id'), primary_key=True)
    odor_threshold_ppb: Mapped[float] = mapped_column(Float, nullable=True)
    odor_potency: Mapped[str] = mapped_column(String, nullable=True)
    russell_valence: Mapped[float] = mapped_column(Float, nullable=True)
    russell_arousal: Mapped[float] = mapped_column(Float, nullable=True)
    evidence_level: Mapped[str] = mapped_column(String, nullable=True)

    ingredient = relationship("Ingredient", back_populates="psychophysics")


class Sustainability(Base):
    __tablename__ = 'sustainability'

    ingredient_id: Mapped[int] = mapped_column(
        ForeignKey('ingredients.id'), primary_key=True)
    biodegradability: Mapped[bool] = mapped_column(Boolean, default=True)
    renewable_source: Mapped[bool] = mapped_column(Boolean, default=True)
    carbon_footprint: Mapped[float] = mapped_column(Float, nullable=True)

    ingredient = relationship("Ingredient", back_populates="sustainability")


def create_all_tables(engine: Engine):
    print(" Criando tabelas no banco de dados...")
    Base.metadata.create_all(engine)
    print(" Tabelas criadas!")
