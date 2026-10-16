import tarfile
with tarfile.open('/content/results038r.tar.gz','w:gz') as t:
    t.add('/content/out038r',arcname='out038r')
