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

bool hkMatrix3d::isOk() const
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
			const hkDouble64& ff = ((&m_col0)[c])(r);
			if( hkMath::isFinite(ff) == false )
			{
				return false;
			}
		}
	}
	return true;
#endif
}

void hkMatrix3d::transpose()
{
	HK_TRANSPOSE3d(m_col0,m_col1,m_col2);
}

void hkMatrix3d_setMulMat3Mat3( hkMatrix3d* THIS, const hkMatrix3d& aTb, const hkMatrix3d& bTc )
{
	HK_ASSERT(0x6d9d1d43,  THIS != &aTb );
	hkVector4dUtil::rotatePoints( aTb, &bTc.getColumn(0), 3, &THIS->getColumn(0) );
}

#if !defined (HK_PLATFORM_SPU)
void hkMatrix3d_invertSymmetric( hkMatrix3d& thisMatrix )
{
	thisMatrix.invertSymmetric();
}
#endif


void hkMatrix3d::setTranspose( const hkMatrix3d& s )
{
	_setTranspose( s );
}

// aTc = aTb * bTc
void hkMatrix3d::setMul( const hkMatrix3d& aTb, const hkMatrix3d& bTc )
{
	HK_ASSERT(0x6d9d1d43,  this != &aTb );
	hkVector4dUtil::rotatePoints( aTb, &bTc.getColumn(0), 3, &m_col0 );
}

void hkMatrix3d::changeBasis(const hkRotationd& r)
{
	hkRotationd temp;
	temp.setMulInverse(*this, r);
	this->setMul(r, temp);
}

void hkMatrix3d::setMulInverse( const hkMatrix3d& aTb, const hkRotationd& cTb )
{
	hkMatrix3dUtil::_computeMulInverse(aTb, cTb, *this);
}

void hkMatrix3d::setMulInverseMul( const hkRotationd& bTa, const hkMatrix3d& bTc )
{
	HK_ASSERT(0xf032e412, this != (hkMatrix3d*)&bTa );
	hkVector4dUtil::rotateInversePoints( bTa, &bTc.getColumn(0), 3, &m_col0 );
}

//
//	Sets this = Transpose(a) * b.

void hkMatrix3d::setTransposeMul(const hkMatrix3d& a, const hkMatrix3d& b)
{
	hkMatrix3dUtil::_computeTransposeMul(a, b, *this);
}

void hkMatrix3d::mul( hkSimdDouble64Parameter scale)
{
	m_col0.mul(scale);
	m_col1.mul(scale);
	m_col2.mul(scale);
}

void hkMatrix3d::setCrossSkewSymmetric( hkVector4dParameter r )
{
#if (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
	hkSimdDouble64 zero; zero.setZero();
	const hkSimdDouble64 r0 = r.getComponent<0>();
	const hkSimdDouble64 r1 = r.getComponent<1>();
	const hkSimdDouble64 r2 = r.getComponent<2>();

	m_col0.set( zero,   r2,  -r1, zero );
	m_col1.set(  -r2, zero,   r0, zero );
	m_col2.set(   r1,  -r0, zero, zero );
#else
	m_col0.set(  hkDouble64(0)   ,  r(2), -r(1) );
	m_col1.set( -r(2),     hkDouble64(0), +r(0) );
	m_col2.set(  r(1), -r(0),     hkDouble64(0) );
#endif
}

void hkMatrix3d::setMul( hkSimdDouble64Parameter scale, const hkMatrix3d& a)
{
	m_col0.setMul(scale, a.getColumn<0>());
	m_col1.setMul(scale, a.getColumn<1>());
	m_col2.setMul(scale, a.getColumn<2>());
}

//
//	Add the product of a and scale (this += a * scale)

void hkMatrix3d::addMul( hkSimdDouble64Parameter scale, const hkMatrix3d& a)
{
	_addMul(scale, a);
}

bool hkMatrix3d::isApproximatelyEqual( const hkMatrix3d& m, hkDouble64 zero) const
{
	hkSimdDouble64 sZ; sZ.setFromFloat(zero);
	return	   m_col0.allEqual<3>( m.getColumn<0>(), sZ )
			&& m_col1.allEqual<3>( m.getColumn<1>(), sZ )
			&& m_col2.allEqual<3>( m.getColumn<2>(), sZ );
}

//
//	Checks if this matrix is equal to m within an optional epsilon.

bool hkMatrix3d::isApproximatelyEqualSimd( const hkMatrix3d& m, hkSimdDouble64Parameter eps) const
{
	return	   m_col0.allEqual<3>( m.getColumn<0>(), eps )
			&& m_col1.allEqual<3>( m.getColumn<1>(), eps )
			&& m_col2.allEqual<3>( m.getColumn<2>(), eps );
}

bool hkMatrix3d::isSymmetric(hkDouble64 epsilon) const
{
	hkMatrix3d T;
	T._setTranspose(*this);
	return this->isApproximatelyEqual(T, epsilon);
}

hkResult hkMatrix3d::invert(hkDouble64 epsilon)
{
	hkVector4d r0; r0.setCross( m_col1, m_col2 );
    hkVector4d r1; r1.setCross( m_col2, m_col0 );
    hkVector4d r2; r2.setCross( m_col0, m_col1 );

    const hkSimdDouble64 determinant = m_col0.dot<3>(r0);
	hkSimdDouble64 absDet; absDet.setAbs(determinant);

	const hkSimdDouble64 eps = hkSimdDouble64::fromFloat(epsilon);
	if( absDet.isGreater(eps * eps * eps) )
	{
		hkSimdDouble64 dinv; dinv.setReciprocal(determinant);
		m_col0.setMul(dinv, r0);
		m_col1.setMul(dinv, r1);
		m_col2.setMul(dinv, r2);
		this->transpose();
		return HK_SUCCESS;
    }

	return HK_FAILURE;
}

void hkMatrix3d::invertSymmetric()
{
	hkVector4d r0; r0.setCross( m_col1, m_col2 );
	hkVector4d r1; r1.setCross( m_col2, m_col0 );
	hkVector4d r2; r2.setCross( m_col0, m_col1 );

	const hkSimdDouble64 eps = hkSimdDouble64_Eps;

	// Compute 1 / determinant. Set it to zero in case of singular matrices!

	const hkSimdDouble64 determinant = m_col0.dot<3>(r0);
	const hkVector4dComparison cmp = determinant.greater(eps*eps*eps);

	// avoid dividing by zero because it throws an exception on some platforms
	hkSimdDouble64 determinantOrOne;
	determinantOrOne.setSelect(cmp, determinant, hkSimdDouble64_1);

	hkSimdDouble64 dInv;
	dInv.setReciprocal(determinantOrOne);
	dInv.zeroIfFalse(cmp);

	m_col0.setMul(r0, dInv);
	m_col1.setMul(r1, dInv);
	m_col2.setMul(r2, dInv);
}

void hkMatrix3d::add( const hkMatrix3d& a )
{
	m_col0.add( a.getColumn<0>() );
	m_col1.add( a.getColumn<1>() );
	m_col2.add( a.getColumn<2>() );
}

void hkMatrix3d::sub( const hkMatrix3d& a )
{
	m_col0.sub( a.getColumn<0>() );
	m_col1.sub( a.getColumn<1>() );
	m_col2.sub( a.getColumn<2>() );
}

void hkMatrix3d::mul( const hkMatrix3d& a )
{
	
	hkMatrix3d temp;
	temp.setMul( *this, a );
	*this = temp;
}

const hkSimdDouble64 hkMatrix3d::getDeterminant() const
{
	hkVector4d r0, r1, r2;
	getRows(r0, r1, r2);

	hkVector4d r1r2;
	r1r2.setCross(r1,r2);
	return r0.dot<3>(r1r2);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Diagonalization of symmetric matrix - based on sec 8.4, Jacobi Methods, of Golub & Van Loan
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct SchurMatrix
{
	hkDouble64 c, s;
	int p, q;
};

static HK_FORCE_INLINE void _constructSchurMatrix(const hkMatrix3d& M, int p, int q, SchurMatrix& S)
{
	S.p = p;
	S.q = q;
	hkDouble64 Mpq = M(p,q);
	if ( Mpq != hkDouble64(0) )
	{
		hkDouble64 tau = ( M(q,q) - M(p,p) ) / (hkDouble64(2) * Mpq);
		hkDouble64 t = hkMath::sqrt(hkDouble64(1) + tau*tau);
		if (tau>=hkDouble64(0))  t = hkDouble64(1) / (tau + t);
		else				 t = hkDouble64(1) / (tau - t);
		S.c = hkMath::sqrtInverse(hkDouble64(1) + t*t);
		S.s = t * S.c;
	}
	else
	{
		S.c = hkDouble64(1);
		S.s = hkDouble64(0);
	}
}

static HK_FORCE_INLINE void _constructSchurMatrixUnchecked(const hkMatrix3d& M, int p, int q, SchurMatrix& S)
{
	S.p = p;
	S.q = q;
	hkDouble64 Mpq = M(p,q);
	hkDouble64 tau = ( M(q,q) - M(p,p) ) / (hkDouble64(2) * Mpq);
	hkDouble64 t = hkMath::sqrt(hkDouble64(1) + tau*tau);
	if (tau>=hkDouble64(0))  t = hkDouble64(1) / (tau + t);
	else				 t = hkDouble64(1) / (tau - t);
	S.c = hkMath::sqrtInverse(hkDouble64(1) + t*t);
	S.s = t * S.c;
}

static HK_FORCE_INLINE const hkSimdDouble64 _frobeniusNormSqr(const hkMatrix3d& M)
{
	return M.getColumn<0>().lengthSquared<3>() + M.getColumn<1>().lengthSquared<3>() + M.getColumn<2>().lengthSquared<3>();
}

hkSimdDouble64 hkMatrix3d::frobeniusNormSqr() const 
{
	return _frobeniusNormSqr(*this);
}


static HK_FORCE_INLINE const hkSimdDouble64 _offDiagNormSqr(const hkMatrix3d& M)
{
	const hkSimdDouble64 m01 = M.getElement<0,1>();
	const hkSimdDouble64 m02 = M.getElement<0,2>();
	const hkSimdDouble64 m12 = M.getElement<1,2>();
	return hkSimdDouble64::getConstant<HK_QUADREAL_2>()*(m01*m01 + m02*m02 + m12*m12);
}

// returns the fabs of the largest off diagonal
static HK_FORCE_INLINE hkDouble64 _findLargestOffDiagEntry(const hkMatrix3d& M, int& p, int& q)
{
	p = 0; q = 1; 
	hkDouble64 maxent = hkMath::fabs(M(0,1)); 
	hkDouble64 mag02  = hkMath::fabs(M(0,2)); 
	hkDouble64 mag12  = hkMath::fabs(M(1,2)); 
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
static HK_FORCE_INLINE void _constructJacobiRotation(hkMatrix3d& J, const SchurMatrix& S)
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


hkResult hkMatrix3d::diagonalizeSymmetric(hkRotationd& eigenVec, hkVector4d& eigenVal, int maxIter, hkDouble64 epsilon) const
{
	HK_ON_DEBUG( if (!isSymmetric(epsilon)) { HK_WARN(0x15e84635, "Attempted to diagonalize an unsymmetric matrix in hkMatrix3d::diagonalizeSymmetric"); return HK_FAILURE; } );
	
	hkMatrix3d M(*this);
	eigenVec.setIdentity();

	hkSimdDouble64 eps; eps.setFromFloat(epsilon);
	const hkSimdDouble64 epsSqr = eps*eps * _frobeniusNormSqr(*this);
	int nIter = 0;
	hkSimdDouble64 normSqr; normSqr.setZero();

	for (;;)
	{
		normSqr = _offDiagNormSqr(M);
		if ( normSqr<=epsSqr || nIter>=maxIter ) break;

		int p = 0, q = 0;
		_findLargestOffDiagEntry(M, p, q);

		SchurMatrix S;
		_constructSchurMatrix(M, p, q, S);

		hkMatrix3d J;  _constructJacobiRotation<false>(J, S);
		hkMatrix3d Jt; _constructJacobiRotation<true>(Jt, S);

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


hkResult hkMatrix3d::diagonalizeSymmetricWarmStart(hkMatrix3d& eigenVec, hkVector4d& eigenVal, int maxIter, hkDouble64 epsilon) const
{
	HK_ON_DEBUG( if (!isSymmetric(epsilon)) { HK_WARN(0x15e84635, "Attempted to diagonalize an unsymmetric matrix in hkMatrix3d::diagonalizeSymmetric"); return HK_FAILURE; } );

	//Warm start, m = eigenVecTrans*m*eigenVec
	hkMatrix3d tempM(*this);
	hkMatrix3d eigenVecTrans(eigenVec);
	eigenVecTrans.transpose();

	hkMatrix3d m; 
	m.setMul(eigenVecTrans, tempM);
	m.mul(eigenVec);
		
	hkSimdDouble64 eps; eps.setFromFloat(epsilon);
	const hkSimdDouble64 epsSqr = eps*eps * _frobeniusNormSqr(*this);
	int nIter = 0;
	hkSimdDouble64 normSqr; normSqr.setZero();

	for (;;)
	{
		normSqr = _offDiagNormSqr(m);
		if ( normSqr<epsSqr || nIter>=maxIter ) break;

		int p = 0, q = 0;
		_findLargestOffDiagEntry(m, p, q);

		SchurMatrix schur;
		_constructSchurMatrix(m, p, q, schur);

		hkMatrix3d jacobi;
		_constructJacobiRotation<false>(jacobi, schur);

		// m <- jacobiTrans*m*jacobi
		hkMatrix3d jacobiTrans(jacobi); jacobiTrans.transpose();
		m.mul(jacobi);
		m.setMul(jacobiTrans,m);

		// eigenVec <- eigenVec*jacobi
		eigenVec.mul(jacobi);

		nIter++;
	}
	
    hkVector4d& col0 = eigenVec.getColumn(0);
	hkVector4d& col1 = eigenVec.getColumn(1);
	hkVector4d& col2 = eigenVec.getColumn(2);

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


void hkMatrix3d::diagonalizeSymmetricApproximation(hkRotationd& eigenVec, hkVector4d& eigenVal, int maxIter) const
{
	hkMatrix3d M(*this);
	eigenVec.setIdentity();

	const hkDouble64 eps = HK_DOUBLE_EPSILON;
	const hkDouble64 epsSqr = eps*eps * _frobeniusNormSqr(*this).getReal();

	for (int nIter = 0; nIter<maxIter; nIter++)
	{
		int p = 0, q = 0;
		hkDouble64 fabsOffDiag = _findLargestOffDiagEntry(M, p, q);

		if ( fabsOffDiag*fabsOffDiag < epsSqr )
		{
			break;
		}

		SchurMatrix S; _constructSchurMatrixUnchecked(M, p, q, S);

		hkMatrix3d J;  _constructJacobiRotation<false>(J, S);
		hkMatrix3d Jt; _constructJacobiRotation<true>(Jt, S);

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
