"""Restore and verify the best-model archive without joining large parts."""
from pathlib import Path, PurePosixPath
import argparse,hashlib,io,json,os,sys,tarfile,time
import zstandard as zstd

class PartsReader(io.RawIOBase):
    def __init__(self,paths):self.paths=iter(paths);self.current=None
    def readable(self):return True
    def read(self,n=-1):
        if n<0:n=8*1024*1024
        chunks=[];remaining=n
        while remaining:
            if self.current is None:
                try:self.current=next(self.paths).open('rb')
                except StopIteration:break
            b=self.current.read(remaining)
            if not b:self.current.close();self.current=None;continue
            chunks.append(b);remaining-=len(b)
        return b''.join(chunks)
    def close(self):
        if self.current:self.current.close()
        super().close()

def verify_and_restore(parts,output,verify_only=False):
    parts=Path(parts);output=Path(output).resolve()
    bundle=json.loads((parts/'bundle-manifest.json').read_text(encoding='utf-8'))
    paths=[]
    for i,part in enumerate(bundle['parts'],1):
        p=parts/part['name'];h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
        if p.stat().st_size!=part['bytes'] or h.hexdigest()!=part['sha256']:
            raise ValueError(f'分卷校验失败: {p.name}')
        paths.append(p);print(f'Parts verified: {i}/{len(bundle["parts"])}',flush=True)
    actual={};archived_manifest=None;last=time.monotonic();read_bytes=0
    with PartsReader(paths) as src, zstd.ZstdDecompressor().stream_reader(src) as stream,tarfile.open(fileobj=stream,mode='r|',bufsize=4*1024*1024) as tf:
        for member in tf:
            if not member.isfile():raise ValueError(f'Unsupported archive entry: {member.name}')
            rel=PurePosixPath(member.name)
            if rel.is_absolute() or '..' in rel.parts or '\\' in member.name:raise ValueError('Unsafe path')
            f=tf.extractfile(member)
            if member.name=='ARCHIVE_FILE_MANIFEST.json':
                blob=f.read()
                if hashlib.sha256(blob).hexdigest()!=bundle['file_manifest_sha256']:raise ValueError('File manifest mismatch')
                archived_manifest=json.loads(blob.decode('utf-8'));continue
            if not rel.parts or rel.parts[0]!='大作业上传':raise ValueError('Unexpected project prefix')
            logical=PurePosixPath(*rel.parts[1:]).as_posix()
            if logical in actual:raise ValueError('Duplicate archive entry')
            destination=output.joinpath(*rel.parts)
            if not destination.resolve().is_relative_to(output):raise ValueError('Unsafe target path')
            writer=None
            if not verify_only:
                destination.parent.mkdir(parents=True,exist_ok=True)
                if destination.exists():raise FileExistsError(f'还原目标已存在: {destination}')
                writer=destination.open('xb')
            h=hashlib.sha256();size=0
            try:
                while True:
                    b=f.read(8*1024*1024)
                    if not b:break
                    h.update(b);size+=len(b);read_bytes+=len(b)
                    if writer:writer.write(b)
                    if time.monotonic()-last>10:
                        print(f'Files processed: {len(actual)}/{bundle["source_files"]}, {read_bytes/1e9:.2f} GB',flush=True);last=time.monotonic()
            finally:
                if writer:writer.close()
            if size!=member.size:raise ValueError('File length mismatch')
            if not verify_only:os.utime(destination,(member.mtime,member.mtime))
            actual[logical]={'bytes':size,'sha256':h.hexdigest()}
        # Consume frame padding and checksum after the tar end marker.
        while stream.read(8*1024*1024):
            pass
    if archived_manifest is None:raise ValueError('Missing archive manifest')
    expected={r['path']:{'bytes':r['bytes'],'sha256':r['sha256']} for r in archived_manifest}
    if actual!=expected:raise ValueError('Restored contents do not match complete manifest')
    if len(actual)!=bundle['source_files'] or read_bytes!=bundle['source_bytes']:raise ValueError('Project totals mismatch')
    print(f'COMPLETE: {len(actual)} files, {read_bytes} bytes verified'+('' if verify_only else f' and restored to {output}'),flush=True)
    return {'files':len(actual),'bytes':read_bytes,'verified':True,'restored':not verify_only}

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    ap=argparse.ArgumentParser();ap.add_argument('--parts',type=Path,default=Path('downloads'));ap.add_argument('--output',type=Path,default=Path('restored'));ap.add_argument('--verify-only',action='store_true')
    a=ap.parse_args();verify_and_restore(a.parts,a.output,a.verify_only)
