from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Text:
    """Сущность текст"""
    x: int
    y: int
    w: int
    h: int
    level: int
    block_num: int
    par_num: int
    line_num: int
    word_num: int
    conf: int
    text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coordinates": (self.x, self.y, self.w, self.h),
            "text": self.text,
            "level": self.level,
            "block_num": self.block_num,
            "par_num": self.par_num,
            "line_num": self.line_num,
            "word_num": self.word_num,
            "conf": self.conf,
        }
@dataclass
class Cell:
    """Сущность ячейки таблицы"""
    x: int
    y: int
    w: int
    h: int
    text: List[Text] = field(default_factory=list)
    column_index: int = -1
    row_index: int = -1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coordinates": (self.x, self.y, self.w, self.h),
            "text": self.text,
            "column": self.column_index,
            "row": self.row_index
        }

@dataclass
class Column:
    """Сущность колонки таблицы"""
    x: int
    cells: List[Cell] = field(default_factory=list)
    _width: int = field(init=False, default=0)
    _avg_x: float = field(init=False, default=0.0)

    def add_cell(self, cell: Cell) -> None:
        """Добавление ячейки с обновлением метрик"""
        self.cells.append(cell)
        self._calculate_metrics()

    def __post_init__(self):
        self._calculate_metrics()

    def _calculate_metrics(self) -> None:
        if self.cells:
            self._width = max(cell.w for cell in self.cells)
            x_coords = [cell.x for cell in self.cells]
            self._avg_x = sum(x_coords) / len(x_coords)

    @property
    def width(self) -> int:
        return self._width

    @property
    def avg_x(self) -> float:
        return self._avg_x

@dataclass
class Row:
    """Сущность строки таблицы"""
    cells: List[Cell] = field(default_factory=list)
    _height: int = field(init=False, default=0)
    _avg_y: float = field(init=False, default=0.0)

    def add_cell(self, cell: Cell) -> None:
        """Добавление ячейки с обновлением метрик"""
        self.cells.append(cell)
        self._calculate_metrics()

    def __post_init__(self):
        self._calculate_metrics()

    def _calculate_metrics(self) -> None:
        if self.cells:
            self._avg_y = min(cell.y for cell in self.cells)
            self._height = max(cell.h for cell in self.cells)
    @property
    def height(self) -> int:
        return self._height

    @property
    def avg_y(self) -> float:
        return self._avg_y

@dataclass
class Table:
    """Сущность таблицы"""
    x: int
    y: int
    w: int
    h: int
    rows: List[Row] = field(default_factory=list)
    columns: List[Column] = field(default_factory=list)

    def add_row(self, cells: List[Cell]) -> None:
        """Группировка ячеек по строкам на основе Y-координаты"""
        sorted_cells = sorted(cells, key=lambda c: c.y)
        grouped_rows: Dict[int, List[Cell]] = {}
        y_tolerance = 5

        for cell in sorted_cells:
            added = False
            for y in grouped_rows.keys():
                if abs(cell.y - y) <= y_tolerance:
                    grouped_rows[y].append(cell)
                    added = True
                    break
            if not added:
                grouped_rows[cell.y] = [cell]

        # Создаем строки из сгруппированных ячеек
        for i, (_, row_cells) in enumerate(sorted(grouped_rows.items())):
            row = Row()
            sorted_row_cells = sorted(row_cells, key=lambda c: c.x)
            for cell in sorted_row_cells:
                cell.row_index = i
                row.add_cell(cell)
            self.rows.append(row)

        self._update_columns()

    def _update_columns(self, x_tolerance: int = 5) -> None:
        """Обновить колонки после добавления ячеек"""
        self.columns.clear()
        all_cells = [cell for row in self.rows for cell in row.cells]
        sorted_cells = sorted(all_cells, key=lambda c: c.x)

        for cell in sorted_cells:
            added = False
            for col in self.columns:
                if abs(cell.x - col.avg_x) <= x_tolerance:
                    col.add_cell(cell)
                    cell.column_index = self.columns.index(col)
                    added = True
                    break
            if not added:
                new_column = Column(cell.x, [cell])
                cell.column_index = len(self.columns)
                self.columns.append(new_column)
    

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def column_count(self) -> int:
        return len(self.columns)