"""Fixed signed hashing of advertisement fields, crosses and short history bags."""
import hashlib
import numpy as np
from scipy.sparse import csr_matrix,vstack

CATEGORIES=['user_id','age','sex','residence','city','city_rank','series_dev','series_group',
    'device_dev','device_name','device_size','net_type','task_id','adv_id','creat_type_cd',
    'adv_prim_id','inter_type_cd','slot_id','site_id','spread_app_id','hispace_app_tags',
    'app_second_class','u_refreshTimes','u_feedLifeCycle']
CROSSES=[('user_id',name) for name in ['task_id','adv_prim_id','spread_app_id','slot_id']]
CROSSES += [(name,target) for name in ['age','sex','city_rank','net_type','series_dev']
            for target in ['task_id','adv_prim_id','spread_app_id','slot_id']]
CROSSES += [(name,'slot_id') for name in ['task_id','adv_prim_id','spread_app_id','app_second_class']]
CHANNELS=[f'ad_{action}_list_v00{i}' for action in ['click','close'] for i in [1,2,3]]+['u_newsCatInterestsST']
FIELDS=CATEGORIES+CHANNELS
WIDTH=1<<20
MASK=(1<<64)-1
A=0x9e3779b97f4a7c15
B=0xbf58476d1ce4e5b9
C=0x94d049bb133111eb


def salt(name):
    return int.from_bytes(hashlib.sha256(name.encode('utf8')).digest()[:8],'little')


def mix(values):
    values=np.asarray(values,dtype='uint64')
    with np.errstate(over='ignore'):
        values=(values^(values>>np.uint64(30)))*np.uint64(B)
        values=(values^(values>>np.uint64(27)))*np.uint64(C)
    return values^(values>>np.uint64(31))


def hashes(name,*values):
    out=np.full(np.asarray(values[0]).shape,salt(name),dtype='uint64')
    for value in values:
        with np.errstate(over='ignore'):
            out=mix(out^(np.asarray(value,dtype='int64').view('uint64')+np.uint64(A)))
    return out


def scalar_hash(name,*values):
    out=salt(name)
    for value in values:
        out^=(int(value)+A)&MASK
        out=((out^(out>>30))*B)&MASK
        out=((out^(out>>27))*C)&MASK
        out^=out>>31
    return out


def tokens(text):
    return sorted(set(int(x) for x in str(text).split('^') if x not in ['','-1','nan','None']))


def encode(frame,width=WIDTH,independent=False):
    if width<2 or width&(width-1):
        raise ValueError('Hash width must be a power of two')
    # Only these input fields are ever inspected; labels and dates are ignored.
    frame=frame[FIELDS].reset_index(drop=True)
    n=len(frame)
    columns=[];rows=[];weights=[]
    numeric={name:frame[name].fillna(-1).to_numpy(dtype='int64') for name in CATEGORIES}
    row_index=np.arange(n,dtype='int32')
    def add(name,values,rr,weight):
        if independent:
            hh=np.array([scalar_hash(name,*vv) for vv in zip(*values)],dtype='uint64')
        else:
            hh=hashes(name,*values)
        columns.append((hh&np.uint64(width-1)).astype('int32'))
        rows.append(rr)
        sign=np.where((hh>>np.uint64(63))==0,1.,-1.).astype('float32')
        weights.append(sign*weight)
    for name in CATEGORIES:
        add('field:'+name,[numeric[name]],row_index,1.)
    for first,second in CROSSES:
        add('cross:'+first+':'+second,[numeric[first],numeric[second]],row_index,1.)
    for channel in CHANNELS:
        strings=frame[channel].fillna('').astype(str)
        cache={text:tokens(text) for text in strings.unique()}
        bags=[cache[text] for text in strings]
        lengths=np.array([len(bag) for bag in bags],dtype='int32')
        if not lengths.sum():
            continue
        rr=np.repeat(row_index,lengths)
        vv=np.concatenate([np.asarray(bag,dtype='int64') for bag in bags])
        scale=(1/np.sqrt(np.maximum(lengths[rr],1))).astype('float32')
        add('history:'+channel,[vv],rr,scale)
        for target in ['spread_app_id','slot_id']:
            add('history_cross:'+channel+':'+target,[vv,numeric[target][rr]],rr,scale)
    matrix=csr_matrix((np.concatenate(weights),(np.concatenate(rows),np.concatenate(columns))),
                      shape=(n,width),dtype='float32')
    matrix.sum_duplicates();matrix.eliminate_zeros();matrix.sort_indices()
    if not np.isfinite(matrix.data).all():
        raise ValueError('Nonfinite sparse features')
    return matrix


def encode_chunks(frame,width=WIDTH,chunk_size=20000,independent=False):
    matrices=[encode(frame.iloc[start:start+chunk_size],width,independent)
              for start in range(0,len(frame),chunk_size)]
    return vstack(matrices,format='csr') if matrices else csr_matrix((0,width),dtype='float32')


def margin(matrix,coefficients,intercept,independent=False):
    coefficients=np.asarray(coefficients,dtype='float64').reshape(-1)
    if coefficients.shape!=(matrix.shape[1],):
        raise ValueError('Coefficient width does not match feature width')
    if not independent:
        return np.asarray(matrix@coefficients).ravel()+float(intercept)
    result=np.full(matrix.shape[0],float(intercept),dtype='float64')
    for start in range(0,matrix.shape[0],20000):
        stop=min(start+20000,matrix.shape[0])
        left,right=matrix.indptr[start],matrix.indptr[stop]
        products=matrix.data[left:right]*coefficients[matrix.indices[left:right]]
        counts=np.diff(matrix.indptr[start:stop+1])
        rr=np.repeat(np.arange(stop-start),counts)
        result[start:stop]+=np.bincount(rr,weights=products,minlength=stop-start)
    return result
