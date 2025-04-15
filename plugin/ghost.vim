" Vim plugin: ghost.vim
if exists("g:loaded_ghost_plugin")
  finish
endif
let g:loaded_ghost_plugin = 1

" Ensure required Vim features (Python3 support and +terminal) are present
if !has('python3')
  echom "Ghost.vim: Python3 support is required"
  finish
endif
if !has('terminal') && !has('nvim')
  echom "Ghost.vim: Terminal support is required (Vim 8+ with +terminal or use Neovim)"
  finish
endif

" Define the :GhostRun command to trigger the autoload function
command! -nargs=0 GhostRun call ghost#Run()
command! -nargs=0 GhostAccept call ghost#Accept()
command! -nargs=0 GhostReject call ghost#Reject()
command! -nargs=0 GhostPrev call ghost#PrevDiff()
command! -nargs=0 GhostNext call ghost#NextDiff()

