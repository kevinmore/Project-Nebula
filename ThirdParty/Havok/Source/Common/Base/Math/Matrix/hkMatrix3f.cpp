/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>
#include <Common/Base/Math/Matrix/hkMatrix3Util.h>

bool hkMatrix3f::isOk() const
{
#if (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
	bool col0Ok = m_col0.isOk<3>();
	bool col1Ok = m_col1.isOk<3>();
	bool col2Ok = m_col2.isOk<3>();
	return col0Ok && col1Ok && col2Ok;
#else
	for(int c=0; c<3; ++c)
	{
		for (int r=0; r<3; ++r)
		{
			const hkFloat32& ff = ((&m_col0)[c])(r);
			if( hkMath::isFinite(ff) == false )
			{
				return false;
			}
		}
	}
	return true;
#endif
}

void hkMatrix3f::transpose()
{
	HK_TRANSPOSE3f(m_col0,m_col1,m_col2);
}

void hkMatrix3f_setMulMat3Mat3( hkMatrix3f* THIS, const hkMatrix3f& aTb, const hkMatrix3f& bTc )
{
	HK_ASSERT(0x6d9d1d43,  THIS != &aTb );
	hkVector4fUtil::rotatePoints( aTb, &bTc.getColumn(0), 3, &THIS->getColumn(0) );
}

#if !defined (HK_PLATFORM_SPU)
void hkMatrix3f_invertSymmetric( hkMatrix3f& thisMatrix )
{
	thisMatrix.invertSymmetric();
}
#endif


void hkMatrix3f::setTranspose( const hkMatrix3f& s )
{
	_setTranspose( s );
}

// aTc = aTb * bTc
void hkMatrix3f::setMul( const hkMatrix3f& aTb, const hkMatrix3f& bTc )
{
	HK_ASSERT(0x6d9d1d43,  this != &aTb );
	hkVector4fUtil::rotatePoints( aTb, &bTc.getColumn(0), 3, &m_col0 );
}

void hkMatrix3f::changeBasis(const hkRotationf& r)
{
	hkRotationf temp;
	temp.setMulInverse(*this, r);
	this->setMul(r, temp);
}

void hkMatrix3f::setMulInverse( const hkMatrix3f& aTb, const hkRotationf& cTb )
{
	hkMatrix3fUtil::_computeMulInverse(aTb, cTb, *this);
}

void hkMatrix3f::setMulInverseMul( const hkRotationf& bTa, const hkMatrix3f& bTc )
{
	HK_ASSERT(0xf032e412, this != (hkMatrix3f*)&bTa );
	hkVector4fUtil::rotateInversePoints( bTa, &bTc.getColumn(0), 3, &m_col0 );
}

//
//	Sets this = Transpose(a) * b.

void hkMatrix3f::setTransposeMul(const hkMatrix3f& a, const hkMatrix3f& b)
{
	hkMatrix3fUtil::_computeTransposeMul(a, b, *this);
}

void hkMatrix3f::mul( hkSimdFloat32Parameter scale)
{
	m_col0.mul(scale);
	m_col1.mul(scale);
	m_col2.mul(scale);
}

void hkMatrix3f::setCrossSkewSymmetric( hkVector4fParameter r )
{
#if (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
	hkSimdFloat32 zero; zero.setZero();
	const hkSimdFloat32 r0 = r.getComponent<0>();
	const hkSimdFloat32 r1 = r.getComponent<1>();
	const hkSimdFloat32 r2 = r.getComponent<2>();

	m_col0.set( zero,   r2,  -r1, zero );
	m_col1.set(  -r2, zero,   r0, zero );
	m_col2.set(   r1,  -r0, zero, zero );
#else
	m_col0.set(  hkFloat32(0)   ,  r(2), -r(1) );
	m_col1.set( -r(2),     hkFloat32(0), +r(0) );
	m_col2.set(  r(1), -r(0),     hkFloat32(0) );
#endif
}

void hkMatrix3f::setMul( hkSimdFloat32Parameter scale, const hkMatrix3f& a)
{
	m_col0.setMul(scale, a.getColumn<0>());
	m_col1.setMul(scale, a.getColumn<1>());
	m_col2.setMul(scale, a.getColumn<2>());
}

//
//	Add the product of a and scale (this += a * scale)

void hkMatrix3f::addMul( hkSimdFloat32Parameter scale, const hkMatrix3f& a)
{
	_addMul(scale, a);
}

bool hkMatrix3f::isApproximatelyEqual( const hkMatrix3f& m, hkFloat32 zero) const
{
	hkSimdFloat32 sZ; sZ.setFromFloat(zero);
	return	   m_col0.allEqual<3>( m.getColumn<0>(), sZ )
			&& m_col1.allEqual<3>( m.getColumn<1>(), sZ )
			&& m_col2.allEqual<3>( m.getColumn<2>(), sZ );
}

//
//	Checks if this matrix is equal to m within an optional epsilon.

bool hkMatrix3f::isApproximatelyEqualSimd( const hkMatrix3f& m, hkSimdFloat32Parameter eps) const
{
	return	   m_col0.allEqual<3>( m.getColumn<0>(), eps )
			&& m_col1.allEqual<3>( m.getColumn<1>(), eps )
			&& m_col2.allEqual<3>( m.getColumn<2>(), eps );
}

bool hkMatrix3f::isSymmetric(hkFloat32 epsilon) const
{
	hkMatrix3f T;
	T._setTranspose(*this);
	return this->isApproximatelyEqual(T, epsilon);
}

hkResult hkMatrix3f::invert(hkFloat32 epsilon)
{
	hkVector4f r0; r0.setCross( m_col1, m_col2 );
    hkVector4f r1; r1.setCross( m_col2, m_col0 );
    hkVector4f r2; r2.setCross( m_col0, m_col1 );

    const hkSimdFloat32 determinant = m_col0.dot<3>(r0);
	hkSimdFloat32 absDet; absDet.setAbs(determinant);

	const hkSimdFloat32 eps = hkSimdFloat32::fromFloat(epsilon);
	if( absDet.isGreater(eps * eps * eps) )
	{
		hkSimdFloat32 dinv; dinv.setReciprocal(determinant);
		m_col0.setMul(dinv, r0);
		m_col1.setMul(dinv, r1);
		m_col2.setMul(dinv, r2);
		this->transpose();
		return HK_SUCCESS;
    }

	return HK_FAILURE;
}

void hkMatrix3f::invertSymmetric()
{
	hkVector4f r0; r0.setCross( m_col1, m_col2 );
	hkVector4f r1; r1.setCross( m_col2, m_col0 );
	hkVector4f r2; r2.setCross( m_col0, m_col1 );

	const hkSimdFloat32 eps = hkSimdFloat32_Eps;

	// Compute 1 / determinant. Set it to zero in case of singular matrices!

	const hkSimdFloat32 determinant = m_col0.dot<3>(r0);
	const hkVector4fComparison cmp = determinant.greater(eps*eps*eps);

	// avoid dividing by zero because it throws an exception on some platforms
	hkSimdFloat32 determinantOrOne;
	determinantOrOne.setSelect(cmp, determinant, hkSimdFloat32_1);

	hkSimdFloat32 dInv;
	dInv.setReciprocal(determinantOrOne);
	dInv.zeroIfFalse(cmp);

	m_col0.setMul(r0, dInv);
	m_col1.setMul(r1, dInv);
	m_col2.setMul(r2, dInv);
}

void hkMatrix3f::add( const hkMatrix3f& a )
{
	m_col0.add( a.getColumn<0>() );
	m_col1.add( a.getColumn<1>() );
	m_col2.add( a.getColumn<2>() );
}

void hkMatrix3f::sub( const hkMatrix3f& a )
{
	m_col0.sub( a.getColumn<0>() );
	m_col1.sub( a.getColumn<1>() );
	m_col2.sub( a.getColumn<2>() );
}

void hkMatrix3f::mul( const hkMatrix3f& a )
{
	
	hkMatrix3f temp;
	temp.setMul( *this, a );
	*this = temp;
}

const hkSimdFloat32 hkMatrix3f::getDeterminant() const
{
	hkVector4f r0, r1, r2;
	getRows(r0, r1, r2);

	hkVector4f r1r2;
	r1r2.setCross(r1,r2);
	return r0.dot<3>(r1r2);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Diagonalization of symmetric matrix - based on sec 8.4, Jacobi Methods, of Golub & Van Loan
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct SchurMatrix
{
	hkFloat32 c, s;
	int p, q;
};

static HK_FORCE_INLINE void _constructSchurMatrix(const hkMatrix3f& M, int p, int q, SchurMatrix& S)
{
	S.p = p;
	S.q = q;
	hkFloat32 Mpq = M(p,q);
	if ( Mpq != hkFloat32(0) )
	{
		hkFloat32 tau = ( M(q,q) - M(p,p) ) / (hkFloat32(2) * Mpq);
		hkFloat32 t = hkMath::sqrt(hkFloat32(1) + tau*tau);
		if (tau>=hkFloat32(0))  t = hkFloat32(1) / (tau + t);
		else				 t = hkFloat32(1) / (tau - t);
		S.c = hkMath::sqrtInverse(hkFloat32(1) + t*t);
		S.s = t * S.c;
	}
	else
	{
		S.c = hkFloat32(1);
		S.s = hkFloat32(0);
	}
}

static HK_FORCE_INLINE void _constructSchurMatrixUnchecked(const hkMatrix3f& M, int p, int q, SchurMatrix& S)
{
	S.p = p;
	S.q = q;
	hkFloat32 Mpq = M(p,q);
	hkFloat32 tau = ( M(q,q) - M(p,p) ) / (hkFloat32(2) * Mpq);
	hkFloat32 t = hkMath::sqrt(hkFloat32(1) + tau*tau);
	if (tau>=hkFloat32(0))  t = hkFloat32(1) / (tau + t);
	else				 t = hkFloat32(1) / (tau - t);
	S.c = hkMath::sqrtInverse(hkFloat32(1) + t*t);
	S.s = t * S.c;
}

static HK_FORCE_INLINE const hkSimdFloat32 _frobeniusNormSqr(const hkMatrix3f& M)
{
	return M.getColumn<0>().lengthSquared<3>() + M.getColumn<1>().lengthSquared<3>() + M.getColumn<2>().lengthSquared<3>();
}

hkSimdFloat32 hkMatrix3f::frobeniusNormSqr() const 
{
	return _frobeniusNormSqr(*this);
}


static HK_FORCE_INLINE const hkSimdFloat32 _offDiagNormSqr(const hkMatrix3f& M)
{
	const hkSimdFloat32 m01 = M.getElement<0,1>();
	const hkSimdFloat32 m02 = M.getElement<0,2>();
	const hkSimdFloat32 m12 = M.getElement<1,2>();
	return hkSimdFloat32::getConstant<HK_QUADREAL_2>()*(m01*m01 + m02*m02 + m12*m12);
}

// returns the fabs of the largest off diagonal
static HK_FORCE_INLINE hkFloat32 _findLargestOffDiagEntry(const hkMatrix3f& M, int& p, int& q)
{
	p = 0; q = 1; 
	hkFloat32 maxent = hkMath::fabs(M(0,1)); 
	hkFloat32 mag02  = hkMath::fabs(M(0,2)); 
	hkFloat32 mag12  = hkMath::fabs(M(1,2)); 
	if ( mag02 > maxent )
	{
		p = 0; q = 2; maxent = mag02;
	}
	if ( mag12 > maxent )
	{
		p = 1; q = 2;
		return mag12;
	}
	return maxent;
}

template<bool transpose>
static HK_FORCE_INLINE void _constructJacobiRotation(hkMatrix3f& J, const SchurMatrix& S)
{
	J.setIdentity();
	J(S.p, S.p) = S.c;
	if ( !transpose)
	{
		J(S.p, S.q) = S.s;
		J(S.q, S.p) = -S.s;
	}
	else
	{
		J(S.p, S.q) = -S.s;
		J(S.q, S.p) = S.s;
	}

	J(S.q, S.q) = S.c;
}


hkResult hkMatrix3f::diagonalizeSymmetric(hkRotationf& eigenVec, hkVector4f& eigenVal, int maxIter, hkFloat32 epsilon) const
{
	HK_ON_DEBUG( if (!isSymmetric(epsilon)) { HK_WARN(0x15e84635, "Attempted to diagonalize an unsymmetric matrix in hkMatrix3f::diagonalizeSymmetric"); return HK_FAILURE; } );
	
	hkMatrix3f M(*this);
	eigenVec.setIdentity();

	hkSimdFloat32 eps; eps.setFromFloat(epsilon);
	const hkSimdFloat32 epsSqr = eps*eps * _frobeniusNormSqr(*this);
	int nIter = 0;
	hkSimdFloat32 normSqr; normSqr.setZero();

	for (;;)
	{
		normSqr = _offDiagNormSqr(M);
		if ( normSqr<=epsSqr || nIter>=maxIter ) break;

		int p = 0, q = 0;
		_findLargestOffDiagEntry(M, p, q);

		SchurMatrix S;
		_constructSchurMatrix(M, p, q, S);

		hkMatrix3f J;  _constructJacobiRotation<false>(J, S);
		hkMatrix3f Jt; _constructJacobiRotation<true>(Jt, S);

		// M <- J^T M J
		M.mul(J);
		M.setMul(Jt,M);

		// V <- V J
		eigenVec.mul(J);

		nIter++;
	}

	eigenVal.set(M.getElement<0,0>(), M.getElement<1,1>(), M.getElement<2,2>(), M.getElement<2,2>());

	if (normSqr>epsSqr)
	{
		return HK_FAILURE;
	}
	return HK_SUCCESS;
}


hkResult hkMatrix3f::diagonalizeSymmetricWarmStart(hkMatrix3f& eigenVec, hkVector4f& eigenVal, int maxIter, hkFloat32 epsilon) const
{
	HK_ON_DEBUG( if (!isSymmetric(epsilon)) { HK_WARN(0x15e84635, "Attempted to diagonalize an unsymmetric matrix in hkMatrix3f::diagonalizeSymmetric"); return HK_FAILURE; } );

	//Warm start, m = eigenVecTrans*m*eigenVec
	hkMatrix3f tempM(*this);
	hkMatrix3f eigenVecTrans(eigenVec);
	eigenVecTrans.transpose();

	hkMatrix3f m; 
	m.setMul(eigenVecTrans, tempM);
	m.mul(eigenVec);
		
	hkSimdFloat32 eps; eps.setFromFloat(epsilon);
	const hkSimdFloat32 epsSqr = eps*eps * _frobeniusNormSqr(*this);
	int nIter = 0;
	hkSimdFloat32 normSqr; normSqr.setZero();

	for (;;)
	{
		normSqr = _offDiagNormSqr(m);
		if ( normSqr<epsSqr || nIter>=maxIter ) break;

		int p = 0, q = 0;
		_findLargestOffDiagEntry(m, p, q);

		SchurMatrix schur;
		_constructSchurMatrix(m, p, q, schur);

		hkMatrix3f jacobi;
		_constructJacobiRotation<false>(jacobi, schur);

		// m <- jacobiTrans*m*jacobi
		hkMatrix3f jacobiTrans(jacobi); jacobiTrans.transpose();
		m.mul(jacobi);
		m.setMul(jacobiTrans,m);

		// eigenVec <- eigenVec*jacobi
		eigenVec.mul(jacobi);

		nIter++;
	}
	
    hkVector4f& col0 = eigenVec.getColumn(0);
	hkVector4f& col1 = eigenVec.getColumn(1);
	hkVector4f& col2 = eigenVec.getColumn(2);

	col0.normalize<3>();
	col1.normalize<3>();
	col2.normalize<3>();
	
	eigenVal.set(m.getElement<0,0>(), m.getElement<1,1>(), m.getElement<2,2>(), m.getElement<2,2>());

	if (normSqr>epsSqr)
	{
		return HK_FAILURE;
	}
	return HK_SUCCESS;
}


void hkMatrix3f::diagonalizeSymmetricApproximation(hkRotationf& eigenVec, hkVector4f& eigenVal, int maxIter) const
{
	hkMatrix3f M(*this);
	eigenVec.setIdentity();

	const hkFloat32 eps = HK_FLOAT_EPSILON;
	const hkFloat32 epsSqr = eps*eps * _frobeniusNormSqr(*this).getReal();

	for (int nIter = 0; nIter<maxIter; nIter++)
	{
		int p = 0, q = 0;
		hkFloat32 fabsOffDiag = _findLargestOffDiagEntry(M, p, q);

		if ( fabsOffDiag*fabsOffDiag < epsSqr )
		{
			break;
		}

		SchurMatrix S; _constructSchurMatrixUnchecked(M, p, q, S);

		hkMatrix3f J;  _constructJacobiRotation<false>(J, S);
		hkMatrix3f Jt; _constructJacobiRotation<true>(Jt, S);

		// M <- J^T M J
		M.mul(J);
		M.setMul(Jt,M);

		// V <- V J
		eigenVec.mul(J);

	}
	eigenVal.set(M.getElement<0,0>(), M.getElement<1,1>(), M.getElement<2,2>(), M.getElement<2,2>());
}

/*
 * Havok SDK - Base file, BUILD(#20130912)
 * 
 * Confidential Information of Havok.  (C) Copyright 1999-2013
 * Telekinesys Research Limited t/a Havok. All Rights Reserved. The Havok
 * Logo, and the Havok buzzsaw logo are trademarks of Havok.  Title, ownership
 * rights, and intellectual property rights in the Havok software remain in
 * Havok and/or its suppliers.
 * 
 * Use of this software for evaluation purposes is subject to and indicates
 * acceptance of the End User licence Agreement for this product. A copy of
 * the license is included with this software and is also available from salesteam@havok.com.
 * 
 */
