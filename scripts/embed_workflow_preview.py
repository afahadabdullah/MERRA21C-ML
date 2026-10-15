#!/usr/bin/env python3
"""Embed an ASCII-only image PDF for the editor's single-source LaTeX compiler.

The normal arXiv build still uses the external vector figure. The fallback uses
the supplied rendered RGB preview PNG, with lossless PNG filters and compression preserved.
"""
import argparse
import base64
from pathlib import Path
import struct

START='% BEGIN INLINE WORKFLOW PREVIEW'
END='% END INLINE WORKFLOW PREVIEW'


def png_pdf(path,width_points=518.4):
    png=Path(path).read_bytes()
    if png[:8]!=b'\x89PNG\r\n\x1a\n':raise ValueError('Expected PNG')
    pos=8;parts=[]
    while pos<len(png):
        size=struct.unpack('>I',png[pos:pos+4])[0];kind=png[pos+4:pos+8];data=png[pos+8:pos+8+size]
        if kind==b'IHDR':w,h,depth,color,_,_,interlace=struct.unpack('>IIBBBBB',data)
        if kind==b'IDAT':parts.append(data)
        pos+=12+size
    if (depth,color,interlace)!=(8,2,0):raise ValueError('Expected noninterlaced 8-bit RGB PNG')
    stream=base64.a85encode(b''.join(parts),wrapcol=100)+b'~>\n'
    height=width_points*h/w
    objects=[b'<< /Type /Catalog /Pages 2 0 R >>',b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
        f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width_points:g} {height:g}] /Resources << /XObject << /Im 4 0 R >> >> /Contents 5 0 R >>'.encode(),
        f'<< /Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter [/ASCII85Decode /FlateDecode] /DecodeParms [null << /Predictor 15 /Colors 3 /BitsPerComponent 8 /Columns {w} >>] /Length {len(stream)} >>\nstream\n'.encode()+stream+b'endstream']
    content=f'q {width_points:g} 0 0 {height:g} 0 0 cm /Im Do Q\n'.encode()
    objects.append(f'<< /Length {len(content)} >>\nstream\n'.encode()+content+b'endstream')
    result=b'%PDF-1.4\n';offsets=[0]
    for i,obj in enumerate(objects,1):
        offsets.append(len(result));result+=f'{i} 0 obj\n'.encode()+obj+b'\nendobj\n'
    xref=len(result);result+=f'xref\n0 {len(objects)+1}\n0000000000 65535 f \n'.encode()
    result+=b''.join(f'{x:010d} 00000 n \n'.encode() for x in offsets[1:])
    result+=f'trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n'.encode()
    return result.decode('ascii')


def embed(tex,png):
    path=Path(tex);source=path.read_text()
    block=START+'\n% External vector PDF is preferred; this lossless fallback keeps the editor preview self-contained.\n'
    block+='\\begin{filecontents*}[overwrite]{MERRA21C_ML_workflow_preview.pdf}\n'+png_pdf(png)
    block+='\\end{filecontents*}\n'+END+'\n'
    if START in source:
        first=source.index(START);last=source.index(END,first)+len(END)
        source=source[:first]+block+source[last:].lstrip('\n')
    else:source=source.replace('\\begin{document}',block+'\\begin{document}',1)
    path.write_text(source)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tex',required=True)
    source=p.add_mutually_exclusive_group(required=True);source.add_argument('--png');source.add_argument('--pdf')
    p.add_argument('--dpi',type=int,default=144)
    a=p.parse_args()
    if a.pdf:
        import pymupdf
        path=Path(a.pdf);a.png=path.with_name(path.stem+'_preview.png')
        with pymupdf.open(path) as doc:doc[0].get_pixmap(dpi=a.dpi,alpha=False).save(a.png)
    embed(a.tex,a.png)
