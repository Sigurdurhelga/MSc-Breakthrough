$URL: https://repository.cs.ru.is/svn/thesis-template/trunk/ruthesis/latex/README.TXT $
$Id: README.TXT 294 2016-03-16 14:40:59Z foley $
README for LaTeX Thesis Templates

Questions, comments, complaints:  latex AT list.ru.is or cs-grad AT ru.is

Many notes are at the top of the DEGREE-NAME-YEAR.tex file, this file
is for a general overview and common problems.  More information on
working with LaTeX at
http://afs.rnd.ru.is/project/htgaru/trunk/how-to-get-around-projects.pdf

Getting started:

0. Find a safe place to work on your thesis.  The author recommends
   Subversion on the projects.cs.ru.is servers, but anywhere data is
   backed up is a good idea.  If you are working with sensitive
   information, you should avoid bitbucket, google drive, dropbox, and
   any other free cloud service.  If you think this is unnecessary,
   just consider how much time you will lose if your computer crashes.
   Due to Murphy's law, this is likely to happen just before your
   thesis is due.

1. Get a LaTeX installation.  We recommend TeXlive https://www.tug.org/texlive/
   For this template on windows, MiKTeX is not recommended because it is missing some of the packages as of this writing.
   You will need to enable the "miktex" option in the template to substitute packages.
   It is very very important that you run the "MikTeX Update Wizard" before you start.  Otherwise you may get errors when you try to build the document.

   Under linux this is the "texlive" package.
   Under Mac/OSX this is the "MacTeX" distribution.

   a) RedHat: sudo yum -y install texlive-collection-fontsrecommended texlive-biblatex-{apa,apa-doc,ieee,ieee-doc} texlive-{xargs,lipsum,lastpage,pseudocode,url,examplep,listings,xspace,pgf,tikz,amsfonts,amsmath,amssymb,siunitx,svn-multi,subfig,fixme,textpos,biblatex,makeglos,nomencl,xwatermark,ltxkeys,framed,boondox,printlen} 
     Getting biber installed on older RedHat systems is a bit tricky for unclear reasons.    
     The metapackage you need is at https://copr.fedoraproject.org/coprs/cbm/Biber/
   b) Debian/Ubuntu:  sudo apt-get -y install texlive-full pgf latex-xcolor
If you don't want to install everything, this list of packages is known to work: 
sudo apt-get -y install texlive texlive-latex-extra texlive-science texlive-generic-extra texlive-lang-european texlive-lang-german latex-xcolor texlive-pictures pgf texlive-bibtex-extra texlive-publishers chktex evince fonts-lmodern lmodern biber

2. Get a LaTeX Integrated Development Environment (recommended, but not required)
   http://texstudio.sourceforge.net/ or
   http://www.xm1math.net/texmaker/
   Some editors may include LaTeX support.
   If you want to learn a very powerful (but old-fashioned) editor http://www.gnu.org/software/emacs/
      Install the auctex package by: M-x list-packages, click on AUCTeX

3. Get a references manager (recommended, but not required)
   http://jabref.sourceforge.net/  (You may have to install a Java JRE first.)
   The reference library is in "main.bib" by default.
   It is just a text file that can be edited, but be careful with the formatting.
   A common mistake is to forget "," at the end of each piece of an entry/line.

   If you are going to make glossaries or acronym lists, you will need
   a perl interpreter.  Only windows usually needs this installed:
   http://www.activestate.com/activeperl

3. Get supporting programs for some tools.
	For glossaries under windows, you will need to install Perl
	http://strawberryperl.com/
	(it is already installed on the other platforms.)

3. Try building the instructions.tex file.  If you get errors, there
   is something wrong with your LaTeX installation.  Fix those first.

4. Rename the DEGREE-NAME-YEAR.tex file with your information.tex  
   e.g. msc-foley-2015.tex.  This is referred to as the "Main" file.

5. Open and read the options at the top of the previous file and set
   it up for your document.  You will need to fill in the title and
   author at least.

6. Start editing all of the .tex files with your content.

7. Compile the document by running pdflatex on the Main file, run the
bibliography tool, then view the result.

8. When you print, make sure that the scale is 100%.  If you allow it
   to resize when printing, the margins won't be right.  If the
   margins aren't right, then the RU logo will not look right on the
   cover.

Final Preparation for PhD
Refer to the directions in the main README.TXT one directory down.

Important Details

- Make absolutely sure that your references.bib is in UTF8.  If it is
another format (CP1251,etc) you may get weird problems with any
accented characters.

- Make sure the rest of the files, particularly the .tex file are in
UTF8 or are at least in the same encoding.  If the files are in
different encoding, you will discover errors with accented characters
when you try to include them together.  Watch out for line endings.
Linux, Windows, and OSX all use different line endings in text files.

- Use biber/biblatex instead of bibtex.
(The template already does this.)
Otherwise Icelandic characters will not work properly in your main.bib file.
TexMaker and TeXStudio require a configuration change to do this.
Refer to the "...Projects" guide above.

- Be consistent about UPPER and lower case in naming files.  Windows
  doesn't care (but some programs in Windows do).  OSX sometimes
  cares.  Linux always cares.

- When using this template with SVN, you will want to tell it to ignore:
*-blx.bib
*.acr
*.acn
*.alg
*.aux
*.bak
*.bbl
*.bcf
*.blg
*.bst
*.dvi
*.glo
*.gl*
*.idx
*.ind
*.ilg
*.ist
*.lo?
*.mw
*.nlo
*.ntn
*.out
*.pdf
*.ps
*.rel
*.run.xml
*.sbl
*.slg
*.snm
*.sym
*.synctex.gz
*.tcp
*.thm
*.tdo
*.to?
*.tmp
*.tmproj
*.xwm
._*
._.DS_Store
.~lock*
auto
Thumbs.db
