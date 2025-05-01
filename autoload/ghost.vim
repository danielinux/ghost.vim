" autoload/ghost.vim
"
"
function! ghost#Run(...) range abort
  if a:firstline != a:lastline
    " Selection case
    let l:abs_path = expand('%:p')
    let l:cwd = getcwd()

    " Ensure file is inside project root
    if l:abs_path[:len(l:cwd)-1] !=# l:cwd
      echom "Ghost error: Selection must be in the current project directory."
      return
    endif

    " Get relative path
    let l:rel_path = substitute(l:abs_path, l:cwd . '/', '', '')

    let l:selection = getline(a:firstline, a:lastline)
    call ghost#OpenMultilinePrompt(l:selection, l:rel_path, a:firstline)
  else
    " No selection
    call ghost#OpenMultilinePrompt()
  endif
endfunction



function! ghost#MaybeComplete() abort
  let l:col = col('.') - 1
  let l:line = getline('.')
  " Get text before the cursor
  let l:prefix = strpart(l:line, 0, l:col)

  " Match the most recent @+ token before the cursor
  if l:prefix =~# '@+[[:alnum:]_./-]*$'
    return "\<C-X>\<C-U>"
  endif
  return "\<Right>"
endfunction

autocmd FileType ghostprompt inoremap <buffer> <expr> <Right> ghost#MaybeComplete()

function! ghost#SubmitMultilineBuffer() abort
  let l:script = ghost#FindScriptPath()
  if empty(l:script)
    echom "ghost.vim: Could not locate run-ghost.py"
    return
  endif
  let l:ghost_win = getbufvar('%', 'ghost_term_win')

  if empty(l:script) || empty(l:ghost_win)
    echom "ghost.vim: internal error — missing vars"
    return
  endif

  let l:lines = getline(1, '$')  " skip header
  let l:input = join(l:lines, "\n")

  call win_gotoid(l:ghost_win)
  " store buffer in .ghost_prompt to reuse later
  call writefile(l:lines, '.ghost_prompt')

  call term_start(['python3', '-u', l:script, l:input], {
        \ 'curwin': v:true,
        \ 'exit_cb': function('ghost#OnExit') })

  "bwipeout!
endfunction

function! ghost#OpenMultilinePrompt(...) abort
  let l:script = ghost#FindScriptPath()
  if empty(l:script)
    echom "ghost.vim: Could not locate run-ghost.py"
    return
  endif

  let l:split_width = 80
  let l:old_splitright = &splitright
  set splitright
  let l:prev_win = win_getid()
  vsplit
  wincmd l
  let l:ghost_win = win_getid()
  execute 'vertical resize ' . l:split_width
  "call win_gotoid(l:prev_win)

  enew
  setlocal buftype=nofile bufhidden=wipe noswapfile
  setlocal filetype=ghostprompt
  setlocal completefunc=ghost#PathComplete
  setlocal modifiable nowrap

  let b:ghost_script = l:script
  let b:ghost_term_win = l:ghost_win
  let l:startline = 1
  if a:0 >= 2
      let b:ghost_source_file = a:2
      let b:ghost_source_line = a:3
      let b:ghost_source_len = len(a:1)
      let l:startline = 2
      let l:tag = '@section:' . b:ghost_source_file . ':' . b:ghost_source_line . ':' . b:ghost_source_len
      echom l:tag
      call setline(1, l:tag)
  endif

  " restore from .ghost_prompt if present
  if filereadable('.ghost_prompt')
    call readfile('.ghost_prompt', 0)
    let l:lines = readfile('.ghost_prompt', 0)
    call setline(l:startline, l:lines)
  endif
  set spell wrap linebreak nolist nonumber
  execute 'command! -buffer GhostSubmit call ghost#SubmitMultilineBuffer()'
  startinsert
endfunction


function! ghost#PathComplete(findstart, base) abort
  if a:findstart
    " Find the position of @+ that starts the token
    let l:line = getline('.')
    let l:col = col('.') - 1

    " Work backward from cursor to locate "@+"
    let l:start = l:col
    while l:start >= 2
      if strpart(l:line, l:start - 2, 2) ==# '@+'
        return l:start - 2
      endif
      let l:start -= 1
    endwhile

    return -1  " Not inside @+ expression
  else
    " Strip off @+ from the text we're completing
    let l:rawpath = substitute(a:base, '^@+', '', '')

    " Use glob() to match files and directories
    let l:matches = glob(l:rawpath . '*', 0, 1)
    let l:items = []

    for path in l:matches
      let l:isdir = isdirectory(path)
      let l:display = fnamemodify(path, ':t') . (l:isdir ? '/' : '')
      let l:insert = '@+' . path . (l:isdir ? '/' : '')
      call add(l:items, {'word': l:insert, 'abbr': l:display})
    endfor

    return l:items
  endif
endfunction

"
function! ghost#Accept() abort
  echom 'Accepting changes...'
  delcommand GhostAccept
  delcommand GhostReject
  delcommand GhostPrev
  delcommand GhostNext
    " Accept the changes from the ghost files
    " copy the contents of the ghost file to the original file
    " and delete the ghost file
    "
  silent! execute 'wincmd h'
  let l:ghost_path = '.ghost/' . expand('%')
  if !filereadable(l:ghost_path)
    echom 'ghost.vim: No ghost file found: ' . l:ghost_path
    return
  endif

  " Replace current buffer with ghost file content
  let l:view = winsaveview()
  silent execute '%delete _'
  silent execute '0read ' . fnameescape(l:ghost_path)
  silent execute 'write!'
  silent! execute 'wincmd l'
  silent! execute 'bwipeout!'
  call winrestview(l:view)
  echom 'Changes accepted.'
endfunction

function! ghost#Reject() abort
  delcommand GhostAccept
  delcommand GhostReject
  delcommand GhostPrev
  delcommand GhostNext
  " Reject the changes from the ghost files
  " Close the right buffer
  let l:ghost_path = '.ghost/' . expand('%')
  silent! execute 'wincmd l'
  silent! execute 'bwipeout!'
  echom 'No changes applied.'
endfunction

function! ghost#ChangedFiles() abort
    " Create a list of all diffs
    let l:diff = system('diff -ru --unidirectional-new-file . .ghost | grep -E "^\\-\\-\\-" | cut -d " " -f 2 | sed -e "s/\t.*//"')
    let l:diffs = split(l:diff, '\n')
    let l:files = []
    " echom 'diff:' . len(l:diffs)

    for line in l:diffs
        " echom 'diff: ' . line
        if !empty(line)
            call add(l:files, line)
        endif
    endfor
    return l:files
endfunction

function! ghost#NextDiff() abort
    " Move to the next diff in the list of changed files
    let l:files = ghost#ChangedFiles()
    " echom 'files:' . len(l:files)


    if len(l:files) > 1
        let g:index = index(l:files, expand('%'))
        if g:index == -1 || g:index >= len(l:files)
            let g:index = 0
        endif
        echom "File " . (g:index + 1) . "of " . len(l:files)
        let l:ghost_file = '.ghost/' . l:files[g:index - 1][2:]
        let l:orig_file = l:files[g:index - 1]
        echom  'files ' . l:orig_file . ' vs ' . l:ghost_file
        silent! execute 'wincmd l'
        silent! execute 'e ' . l:ghost_file
        silent! execute 'diffthis'
        silent! execute 'wincmd h'
        silent! execute 'e ' . l:orig_file
        silent! execute 'diffthis | wincmd l'
    endif
endfunction

function! ghost#PrevDiff() abort
    " Move to the prev diff in the list of changed files
    let l:files = ghost#ChangedFiles()

    if len(l:files) > 1
        let g:index = index(l:files, expand('%'))
        if g:index <= 0
            let g:index = len(l:files) - 1
        endif
        let l:ghost_file = '.ghost/' . l:files[g:index - 1][2:]
        let l:orig_file = l:files[g:index - 1]
        silent! execute 'wincmd l'
        silent! execute 'e ' . l:ghost_file
        silent! execute 'diffthis'
        silent! execute 'wincmd h'
        silent! execute 'e ' . l:orig_file
        silent! execute 'diffthis'
        silent! execute 'wincmd l'
    endif
endfunction

function! ghost#OnExit(job, exit_code) abort
  " Find mirror file to the one open in the current buffer in
  " ./.ghost/path/to/file
  let l:ghost_path = '.ghost/' . expand('%')
  echom l:ghost_path
  execute 'diffthis | set splitright | vnew ' . l:ghost_path . '| diffthis'

  let l:files = ghost#ChangedFiles()

  " if there are no diffs, reject the changes
  if empty(l:files)
      call ghost#Reject()
      return
  else
      command! -nargs=0 GhostAccept call ghost#Accept()
      command! -nargs=0 GhostReject call ghost#Reject()
      command! -nargs=0 GhostPrev call ghost#PrevDiff()
      command! -nargs=0 GhostNext call ghost#NextDiff()
      " Wait for user to press ENTER
      call input('Press ENTER to show the proposed changes')
      if v:char == "\<Esc>"
          call ghost#Reject()
          return
      endif

      " Get terminal buffer from job and close it
      let l:bufnr = ch_getbufnr(a:job, 'out')
      if l:bufnr != -1 && bufexists(l:bufnr)
        silent! execute 'bwipeout! ' . l:bufnr
      endif
      " if the current buffer is in the list, the diff is already on the screen.
      if index(l:files, expand('%')) != -1
          return
      endif

      " Open the first file in the list and switch to it
      let l:first_ghost_file = '.ghost/' . l:files[0]
      let l:first_orig_file = l:files[0]
      silent! execute 'wincmd l'
      silent! execute 'e ' . l:first_ghost_file
      silent! execute 'diffthis'
      silent! execute 'wincmd h'
      silent! execute 'e ' . l:first_orig_file
      silent! execute 'diffthis | wincmd l'
  endif
endfunction

function! ghost#FindScriptPath() abort
  for dir in split(&runtimepath, ',')
    let l:path = dir . '/python/run-ghost.py'
    if filereadable(l:path)
      return l:path
    endif
  endfor
  return ''
endfunction

