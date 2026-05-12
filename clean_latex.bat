@echo off
echo Cleaning up LaTeX auxiliary files in the current directory...

:: Delete common auxiliary files
del /q *.aux *.log *.toc *.lof *.lot *.bbl *.blg *.out *.fdb_latexmk *.fls *.synctex.gz *.nav *.snm *.vrb *.maf *.mtc *.mtc0 2>nul

echo Cleanup complete!
pause
