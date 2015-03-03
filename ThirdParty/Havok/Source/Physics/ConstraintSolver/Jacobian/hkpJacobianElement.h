/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKP_JACOBIAN_ELEMENT_H
#define HKP_JACOBIAN_ELEMENT_H

#include <Physics/ConstraintSolver/Accumulator/hkpVelocityAccumulator.h>


///
class hkp2AngJacobian
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_CONSTRAINT_SOLVER, hkp2AngJacobian );

		/// the angular part of the constraint
		//  angular[0].w component is the invJacDiag
		hkVector4	m_angular[2];


		HK_FORCE_INLINE hkReal getInvJacDiag() const					{	return m_angular[0](3);					}
		HK_FORCE_INLINE hkSimdReal getInvJacDiagSr() const				{	return m_angular[0].getComponent<3>();	}
		HK_FORCE_INLINE void  setInvJacDiag( hkReal v )					{	m_angular[0](3) = v;					}
		HK_FORCE_INLINE void  setInvJacDiag( hkSimdRealParameter v )	{	m_angular[0].setComponent<3>(v);		}

		HK_FORCE_INLINE void  setAngularRhs( hkReal v )					{	m_angular[1](3) = v;					}
		HK_FORCE_INLINE void  setAngularRhs( hkSimdRealParameter v )	{	m_angular[1].setComponent<3>(v);		}
		HK_FORCE_INLINE hkReal getAngularRhs( ) const					{	return m_angular[1](3);					}
		HK_FORCE_INLINE hkSimdReal getAngularRhsSr( ) const				{	return m_angular[1].getComponent<3>();	}

	//private:

			// get the diag where |linear| = 0.0f and |angular| = 1.0f
		HK_FORCE_INLINE hkSimdReal getAngularDiag( const hkpVelocityAccumulator& mA, const hkpVelocityAccumulator& mB ) const;

		HK_FORCE_INLINE hkSimdReal getNonDiagOptimized( const hkpVelocityAccumulator& mA, const hkpVelocityAccumulator& mB, const hkp2AngJacobian& jacB ) const;
		HK_FORCE_INLINE hkSimdReal getNonDiagSameObjects( const hkpVelocityAccumulator& mA, const hkpVelocityAccumulator& mB, const hkp2AngJacobian& jacB ) const;
		HK_FORCE_INLINE hkSimdReal getNonDiagDifferentObjects_With2ndBodyFromFirstObject( const hkpVelocityAccumulator& mA, const hkp2AngJacobian& jacB ) const;
};


///
class hkp1Lin2AngJacobian
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_CONSTRAINT_SOLVER, hkp1Lin2AngJacobian );

			/// the linear part
		hkVector4	m_linear0;		// .w component is the rhs

			/// the angular part of the constraint
		// angular[0].w component is the invJacDiag
		// angular[1].w usage is dependent on the hkpJacobianSchema
		hkVector4	m_angular[2];


			// This casts the jacobian to a hk2Ang jacobian. Note that you cannot access its Rhs or AngularRhs values.
			// This requires that body hk2AngJac's and hk1Lin2AngJac's getInvJacDiag() use the same variable in m_angular vectors .
		HK_FORCE_INLINE hkp2AngJacobian& as2AngJacobian_internal() { return *reinterpret_cast<hkp2AngJacobian*>(&m_angular[0]); }

		// warning: call the next function only after setting the linear and angular part
		HK_FORCE_INLINE void setRHS( hkReal v )					{	m_linear0(3) = v;					}
		HK_FORCE_INLINE void setRHS( hkSimdRealParameter v )	{	m_linear0.setComponent<3>(v);		}
		HK_FORCE_INLINE hkReal& getRHS()						{	return m_linear0(3);				}
		HK_FORCE_INLINE const hkReal& getRHS() const			{	return m_linear0(3);				}
		HK_FORCE_INLINE const hkSimdReal getRhsSr() const		{	return m_linear0.getComponent<3>();	}

		HK_FORCE_INLINE hkReal getInvJacDiag() const				{	return m_angular[0](3);					}
		HK_FORCE_INLINE hkSimdReal getInvJacDiagSr() const			{	return m_angular[0].getComponent<3>();	}
		HK_FORCE_INLINE void setInvJacDiag( hkReal v )				{	m_angular[0](3) = v;					}
		HK_FORCE_INLINE void setInvJacDiag( hkSimdRealParameter v )	{	m_angular[0].setComponent<3>(v);		}

		HK_FORCE_INLINE hkReal getUserValue() const					{	return m_angular[1](3);					}
		HK_FORCE_INLINE hkSimdReal getUserValueSr() const			{	return m_angular[1].getComponent<3>();	}
		HK_FORCE_INLINE void setUserValue( hkReal v )				{	m_angular[1](3) = v;					}
		HK_FORCE_INLINE void setUserValue( hkSimdRealParameter v )	{	m_angular[1].setComponent<3>(v);		}

	//private:

			// Get J dot ((M-1)*J)  restrictions: |linear| = 1.0f
		HK_FORCE_INLINE hkSimdReal getDiag( const hkpVelocityAccumulator& mA, const hkpVelocityAccumulator& mB ) const;

			/// Get the non diagonal element of the 2*2 inverse mass matrix in the case that jacA and jacB share exactly the same rigid bodies.
			/// Gets J dot ((M-1)*JacB).
			/// This is a special implementation which makes use of the fact that some PlayStation(R)2 implementations left the last
			/// Jacobian in registers. Be extra careful when using this function.
		HK_FORCE_INLINE hkSimdReal getNonDiag( const hkpVelocityAccumulator& mA, const hkpVelocityAccumulator& mB, const hkp1Lin2AngJacobian& jacB ) const;

			/// Get the non diagonal element in the case that jacA and jacB share exactly the same rigid bodies.
			/// Gets J dot ((M-1)*JacB).
			/// If this and jacB are connecting the same object pair in the same direction.
		HK_FORCE_INLINE hkSimdReal getNonDiagSameObjects( const hkpVelocityAccumulator& mA, const hkpVelocityAccumulator& mB, const hkp1Lin2AngJacobian& jacB ) const;

			/// Get the non diagonal element in the case of different rigid bodies.
			/// Gets J dot ((M-1)*JacB).
			/// Given that mA is the common object of both Jacobians and the mB differ.
		HK_FORCE_INLINE hkSimdReal getNonDiagDifferentObjects( const hkpVelocityAccumulator& mA, const hkp1Lin2AngJacobian& jacB ) const;

		// get J dot ((M-1)*JacB)
		// given that mB of this jacobian is mA of jacB
		HK_FORCE_INLINE hkSimdReal getNonDiagDifferentObjects_With2ndBodyFromFirstObject( const hkpVelocityAccumulator& mA, const hkp1Lin2AngJacobian& jacB ) const;


			// Compute J dot ((M-1)*J)  restrictions: |linear| = 1.0f
		template<class VEL>
		static HK_FORCE_INLINE hkSimdReal computeDiag( hkVector4Parameter jacLinA, hkVector4Parameter jacAngA, hkVector4Parameter jacAngB,
														const VEL& velAccA, const VEL& velAccB );
};


///
class hkpJacDouble2Bil
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_CONSTRAINT_SOLVER, hkpJacDouble2Bil );

	    hkp1Lin2AngJacobian  m_jac0;
	    hkp1Lin2AngJacobian  m_jac1;
};


///
class hkpJacTriple2Bil1Ang
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_CONSTRAINT_SOLVER, hkpJacTriple2Bil1Ang );

	    hkp1Lin2AngJacobian  m_jac0;
	    hkp1Lin2AngJacobian  m_jac1;
	    hkp2AngJacobian		m_jac2;
};


///
class hkp2Lin2AngJacobian
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_CONSTRAINT_SOLVER, hkp2Lin2AngJacobian );

			/// the linear part. m_linear[0].w component is the rhs
		hkVector4   m_linear[2];

			/// the angular part of the constraint
			// angular[0].w component is the invJacDiag
			// angular[1].w usage is dependent on the hkpJacobianSchema
		hkVector4	m_angular[2];


		// warning: call the next function only after setting the linear and angular part
		HK_FORCE_INLINE void setRHS( hkReal v )					{	m_linear[0](3) = v;						}
		HK_FORCE_INLINE void setRHS( hkSimdRealParameter v )	{	m_linear[0].setComponent<3>(v);			}
		HK_FORCE_INLINE hkReal getRHS() const					{	return m_linear[0](3);					}
		HK_FORCE_INLINE hkSimdReal getRhsSr() const				{	return m_linear[0].getComponent<3>();	}

		HK_FORCE_INLINE hkReal getInvJacDiag() const				{	return m_angular[0](3);					}
		HK_FORCE_INLINE hkSimdReal getInvJacDiagSr() const			{	return m_angular[0].getComponent<3>();	}
		HK_FORCE_INLINE void setInvJacDiag( hkReal v )				{	m_angular[0](3) = v;					}
		HK_FORCE_INLINE void setInvJacDiag( hkSimdRealParameter v )	{	m_angular[0].setComponent<3>(v);		}

	//private:

			// get J dot ((M-1)*J)  restrictions: |linear| = 1.0f
		HK_FORCE_INLINE hkSimdReal getDiag( const hkpVelocityAccumulator& mA, const hkpVelocityAccumulator& mB, hkSimdRealParameter leverageRatio ) const;
};


#include <Physics/ConstraintSolver/Jacobian/hkpJacobianElement.inl>


#endif // HKP_JACOBIAN_ELEMENT_H

/*
 * Havok SDK - Product file, BUILD(#20130912)
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
