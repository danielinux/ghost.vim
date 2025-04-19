" autoload/ghost.vim

function! ghost#Run() abort
  if &modified
      echom "Buffer has unsaved changes. Please save your changes before summoning ghosts."
      return
  endif
  " Prompt the user for input text
  let l:input_text = input("Ghost >>: ")
  if empty(l:input_text)
    return
  endif

  " Find run-ghost.py in any plugin directory under runtimepath
  let l:script_path = ghost#FindScriptPath()
  if empty(l:script_path)
    echom "ghost.vim: Could not locate run-ghost.py in runtimepath"
    return
  endif

  " Save current window to return to after starting the terminal job
  let l:prev_win = win_getid()
  let l:split_width = 80
  " let l:old_splitbelow = &splitbelow
  " set splitbelow
  let l:old_splitright = &splitright
  set splitright

  " Open vertical terminal split on the right
  vsplit
  wincmd l
  let l:ghost_win = win_getid()
  execute 'vertical resize' . l:split_width
  call term_start(['python3', '-u', l:script_path, l:input_text], {
        \ 'curwin': v:true,
        \ 'exit_cb': function('ghost#OnExit') })

  " Restore 'splitbelow' setting
  " let &splitbelow = l:old_splitbelow
  let &splitright = l:old_splitright

  " Return to original window
  " call win_gotoid(l:prev_win)
  call win_gotoid(l:ghost_win)
endfunction

function! ghost#Accept() abort
  echom 'Accepting changes...'
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
    " Reject the changes from the ghost files
    " Close the right buffer
    let l:ghost_path = '.ghost/' . expand('%')
    silent! execute 'wincmd l'
    silent! execute 'bwipeout!'
    echom 'No changes applied.'
endfunction
  
function! ghost#ChangedFiles() abort
    " Create a list of all diffs
    let l:diff = system('diff -ru . .ghost | grep -E "^\\-\\-\\-" | cut -d " " -f 2 | sed -e "s/\t.*//"')
    let l:diffs = split(l:diff, '\n')
    let l:files = []
    " echom 'diff:' . len(l:diffs)
  
    for line in l:diffs
        echom 'diff: ' . line
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
        let l:index = index(l:files, expand('%'))
        if l:index != -1 && l:index < len(l:files) - 1
            let l:ghost_file = '.ghost/' . l:files[l:index - 1]
            let l:orig_file = '.ghost/' . l:files[l:index - 1]
            silent! execute 'wincmd l | e ' . l:ghost_file
            silent! execute 'diffthis'
            silent! execute 'wincmd h | e ' . l:orig_file 
            silent! execute 'diffthis | wincmd l'
        endif
    endif
endfunction

function! ghost#PrevDiff() abort
    " Move to the prev diff in the list of changed files
    let l:files = ghost#ChangedFiles()

    if len(l:files) > 1
        let l:index = index(l:files, expand('%'))
        if l:index != -1 && l:index > 0
            let l:ghost_file = '.ghost/' . l:files[l:index - 1]
            let l:orig_file = '.ghost/' . l:files[l:index - 1]
            silent! execute 'wincmd l | e ' . l:ghost_file
            silent! execute 'diffthis'
            silent! execute 'wincmd h | e ' . l:orig_file 
            silent! execute 'diffthis | wincmd l'
        endif
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

