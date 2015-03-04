/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/Math/Vector/hkHalf8Util.h>
#include <Common/Base/Math/Vector/Mx/hkMxUnroll.h>

namespace hknpSolverVelocity_Implementation
{
	template<typename VELOCITY, typename MXVECTOR>
	struct GetAngularVelocityAction1
	{
		const VELOCITY*	m_velocities;
		MXVECTOR&		m_angVelsOut;

		HK_FORCE_INLINE GetAngularVelocityAction1(const VELOCITY* velocities, MXVECTOR& angVelsOut)
		: m_velocities(velocities), m_angVelsOut(angVelsOut) {}

		template <int I> HK_FORCE_INLINE void step()
		{
			m_velocities[I].getAngularVelocity( m_angVelsOut.m_vec.v[I]);
		}
	};

	template<typename VELOCITY, typename MXVECTOR>
	struct GetAngularVelocityAction2
	{
		const VELOCITY**	m_velocities;
		MXVECTOR&			m_angVelsOut;

		HK_FORCE_INLINE GetAngularVelocityAction2(const VELOCITY** velocities, MXVECTOR& angVelsOut)
		: m_velocities(velocities), m_angVelsOut(angVelsOut) {}

		template <int I> HK_FORCE_INLINE void step()
		{
			m_velocities[I]->getAngularVelocity( m_angVelsOut.m_vec.v[I]);
		}
	};

	template<typename VELOCITY, typename MXVECTOR>
	struct GetVelocitiesAction1
	{
		const VELOCITY*	m_velocities;
		MXVECTOR&		m_linVelsOut;
		MXVECTOR&		m_angVelsOut;

		HK_FORCE_INLINE GetVelocitiesAction1(const VELOCITY* velocities, MXVECTOR& linVelsOut, MXVECTOR& angVelsOut)
		: m_velocities(velocities), m_linVelsOut(linVelsOut), m_angVelsOut(angVelsOut) {}

		template <int I> HK_FORCE_INLINE void step()
		{
			m_velocities[I].getLinearVelocity(m_linVelsOut.m_vec.v[I]);
			m_velocities[I].getAngularVelocity(m_angVelsOut.m_vec.v[I]);
		}
	};

	template<typename VELOCITY, typename MXVECTOR>
	struct GetVelocitiesAction2
	{
		const VELOCITY**m_velocities;
		MXVECTOR&		m_linVelsOut;
		MXVECTOR&		m_angVelsOut;

		HK_FORCE_INLINE GetVelocitiesAction2(const VELOCITY** velocities, MXVECTOR& linVelsOut, MXVECTOR& angVelsOut)
		: m_velocities(velocities), m_linVelsOut(linVelsOut), m_angVelsOut(angVelsOut) {}

		template <int I> HK_FORCE_INLINE void step()
		{
			m_velocities[I]->getLinearVelocity(m_linVelsOut.m_vec.v[I]);
			m_velocities[I]->getAngularVelocity(m_angVelsOut.m_vec.v[I]);
		}
	};

	template<typename VELOCITY, typename MXVECTOR>
	struct SetVelocitiesAction1
	{
		VELOCITY*		m_velocitiesOut;
		const MXVECTOR&	m_linVels;
		const MXVECTOR&	m_angVels;

		HK_FORCE_INLINE SetVelocitiesAction1(VELOCITY* velocitiesOut, const MXVECTOR& linVels, const MXVECTOR& angVels)
		: m_velocitiesOut(velocitiesOut), m_linVels(linVels), m_angVels(angVels) {}

		template <int I> HK_FORCE_INLINE void step()
		{
			m_velocitiesOut[I].setVelocity(m_linVels.template getVector<I>(), m_angVels.template getVector<I>());
		}
	};

	template<typename VELOCITY, typename MXVECTOR>
	struct SetVelocitiesAction2
	{
		VELOCITY**		m_velocitiesOut;
		const MXVECTOR&	m_linVels;
		const MXVECTOR&	m_angVels;

		HK_FORCE_INLINE SetVelocitiesAction2(VELOCITY** velocitiesOut, const MXVECTOR& linVels, const MXVECTOR& angVels)
		: m_velocitiesOut(velocitiesOut), m_linVels(linVels), m_angVels(angVels) {}

		template <int I> HK_FORCE_INLINE void step()
		{
			m_velocitiesOut[I]->setVelocity(m_linVels.template getVector<I>(), m_angVels.template getVector<I>());
		}
	};

	template<typename VELOCITY, typename MXVECTOR>
	struct GetInvInertiasAction1
	{
		const VELOCITY**m_velocities;
		MXVECTOR&		m_invInertiasOut;

		HK_FORCE_INLINE GetInvInertiasAction1(const VELOCITY** velocities, MXVECTOR& invInertiasOut)
			: m_velocities(velocities), m_invInertiasOut(invInertiasOut) {}

		template <int I> HK_FORCE_INLINE void step()
		{
			m_velocities[I]->getInvInertias(m_invInertiasOut.m_vec.v[I]);
		}
	};

	template<typename VELOCITY, typename MXVECTOR>
	struct GetAllAction2
	{
		const VELOCITY**m_velocities;
		MXVECTOR&		m_linVelsOut;
		MXVECTOR&		m_angVelsOut;
		MXVECTOR&		m_invInertiasOut;

		HK_FORCE_INLINE GetAllAction2(const VELOCITY** velocities, MXVECTOR& linVelsOut, MXVECTOR& angVelsOut, MXVECTOR& invInertiasOut)
			: m_velocities(velocities), m_linVelsOut(linVelsOut), m_angVelsOut(angVelsOut), m_invInertiasOut(invInertiasOut) {}

		template <int I> HK_FORCE_INLINE void step()
		{
			m_velocities[I]->getLinearVelocity(m_linVelsOut.m_vec.v[I]);
			m_velocities[I]->getAngularVelocity(m_angVelsOut.m_vec.v[I]);
			m_velocities[I]->getInvInertias(m_invInertiasOut.m_vec.v[I]);
		}
	};

	struct Masks
	{
		static HK_ALIGN16( const hkUint32 SetShiftMask[4] );
		static HK_ALIGN16( const hkUint32 SetPermMask[4] );
		static HK_ALIGN16( const hkUint32 GetAngPermMask[4] );
	};
}


HK_FORCE_INLINE void hknpSolverVelocity::init()
{
	vector0()->setZero();
	vector1()->setZero();
}



HK_FORCE_INLINE void hknpSolverVelocity::setLinearVelocity(hkVector4Parameter v)
{
	linearVelocity()->setSelect<hkVector4ComparisonMask::MASK_XYZ>(v, *linearVelocity());
}


HK_FORCE_INLINE void hknpSolverSumVelocity::setLinearVelocity(hkVector4Parameter v)
{
	linearVelocity()->setSelect<hkVector4ComparisonMask::MASK_XYZ>(v, *linearVelocity());
}


template<typename MXVECTOR>
/*static */HK_FORCE_INLINE void	hknpSolverVelocity::getLinearVelocities(const hknpSolverVelocity* velocities, MXVECTOR& linVelsOut)
{
	linVelsOut.template gather<sizeof(hknpSolverVelocity)>(velocities->linearVelocity());
}


template<typename MXVECTOR>
/* static */ HK_FORCE_INLINE void	hknpSolverVelocity::setLinearVelocities(const MXVECTOR& linVels, hknpSolverVelocity* velsOut)
{
	linVels.template scatter<sizeof(hknpSolverVelocity), 3>(velsOut->linearVelocity());
}


template<typename MXVECTOR>
/* static */ HK_FORCE_INLINE void hknpSolverSumVelocity::setLinearVelocities(const MXVECTOR& linVels, hknpSolverSumVelocity* velsOut)
{
	linVels.template scatter<sizeof(hknpSolverSumVelocity), 3>(velsOut->linearVelocity());
}


template<typename MXVECTOR>
/* static */ HK_FORCE_INLINE void	hknpSolverVelocity::getLinearVelocities(const hknpSolverVelocity* velocities[], MXVECTOR& linVelsOut)
{
	enum { OFFSET = HK_OFFSET_OF(hknpSolverVelocity, m_linearVelocity) };
	linVelsOut.template gatherWithOffset<OFFSET>( (const void**)velocities);
}


template<typename MXVECTOR>
/* static */ HK_FORCE_INLINE void hknpSolverSumVelocity::getLinearVelocities(const hknpSolverSumVelocity* velocities[], MXVECTOR& linVelsOut)
{
	enum { OFFSET = HK_OFFSET_OF(hknpSolverSumVelocity, m_linearVelocity) };
	linVelsOut.template gatherWithOffset<OFFSET>( (const void**)velocities);
}


template<typename MXVECTOR>
/* static */ HK_FORCE_INLINE void	hknpSolverVelocity::setLinearVelocities(const MXVECTOR& linVels, hknpSolverVelocity** velsOut)
{
	enum { OFFSET = HK_OFFSET_OF(hknpSolverVelocity, m_linearVelocity) };
	linVels.template scatterWithOffset<OFFSET, 3>((void**) velsOut);
}


template<typename MXVECTOR>
/* static */ HK_FORCE_INLINE void hknpSolverSumVelocity::getLinearVelocities(const hknpSolverSumVelocity* velocities, MXVECTOR& linVelsOut)
{
	linVelsOut.template gather<sizeof(hknpSolverSumVelocity)>(velocities->linearVelocity());
}


template<typename MXVECTOR>
/* static */ HK_FORCE_INLINE void hknpSolverSumVelocity::setLinearVelocities(const MXVECTOR& linVels, hknpSolverSumVelocity** velsOut)
{
	enum { OFFSET = HK_OFFSET_OF(hknpSolverSumVelocity, m_linearVelocity) };
	linVels.template scatterWithOffset<OFFSET, 3>((void**) velsOut);
}


HK_FORCE_INLINE void hknpSolverVelocity::setAngularVelocity(hkVector4Parameter v)
{
	v.store<3, HK_IO_NATIVE_ALIGNED>(angularVelocity());
}


template<typename MXVECTOR>
/* static */ HK_FORCE_INLINE void	hknpSolverVelocity::getAngularVelocities(const hknpSolverVelocity* velocities, MXVECTOR& angVelsOut)
{
	hknpSolverVelocity_Implementation::GetAngularVelocityAction1<hknpSolverVelocity, MXVECTOR> action(velocities, angVelsOut);
	hkMxUnroller<0, MXVECTOR::mxLength>::step(action);
}


template<typename MXVECTOR>
/* static */ HK_FORCE_INLINE void hknpSolverVelocity::setAngularVelocities(const MXVECTOR& angVels, hknpSolverVelocity* velsOut)
{
	angVels.template store<sizeof(hknpSolverVelocity), 3, HK_IO_NATIVE_ALIGNED>(velsOut->angularVelocity());
}


template<typename MXVECTOR>
/* static */ HK_FORCE_INLINE void	hknpSolverVelocity::getAngularVelocities(const hknpSolverVelocity* velocities[], MXVECTOR& angVelsOut)
{
	hknpSolverVelocity_Implementation::GetAngularVelocityAction2<hknpSolverVelocity, MXVECTOR> action(velocities, angVelsOut);
	hkMxUnroller<0, MXVECTOR::mxLength>::step(action);
}


template<typename MXVECTOR>
/* static */ HK_FORCE_INLINE void hknpSolverVelocity::setAngularVelocities(const MXVECTOR& angVels, hknpSolverVelocity** velsOut)
{
	enum { OFFSET = HK_OFFSET_OF(hknpSolverVelocity, m_angularVelocity) };
	angVels.template storeWithOffset<OFFSET, 3, HK_IO_NATIVE_ALIGNED>((AngularVelocityDataType**) velsOut);
}


template<typename MXVECTOR>
/* static */ HK_FORCE_INLINE void	hknpSolverSumVelocity::getAngularVelocities(const hknpSolverSumVelocity* velocities, MXVECTOR& angVelsOut)
{
	hknpSolverVelocity_Implementation::GetAngularVelocityAction1<hknpSolverSumVelocity, MXVECTOR> action(velocities, angVelsOut);
	hkMxUnroller<0, MXVECTOR::mxLength>::step(action);
}


template<typename MXVECTOR>
/* static */ HK_FORCE_INLINE void hknpSolverSumVelocity::setAngularVelocities(const MXVECTOR& angVels, hknpSolverSumVelocity* velsOut)
{
	angVels.template store<sizeof(hknpSolverSumVelocity), 3, HK_IO_NATIVE_ALIGNED>(velsOut->angularVelocity());
}


template<typename MXVECTOR>
/* static */ HK_FORCE_INLINE void	hknpSolverSumVelocity::getAngularVelocities(const hknpSolverSumVelocity* velocities[], MXVECTOR& angVelsOut)
{
	hknpSolverVelocity_Implementation::GetAngularVelocityAction2<hknpSolverSumVelocity, MXVECTOR> action(velocities, angVelsOut);
	hkMxUnroller<0, MXVECTOR::mxLength>::step(action);
}


template<typename MXVECTOR>
/* static */ HK_FORCE_INLINE void hknpSolverSumVelocity::setAngularVelocities(const MXVECTOR& angVels, hknpSolverSumVelocity** velsOut)
{
	enum { OFFSET = HK_OFFSET_OF(hknpSolverSumVelocity, m_angularVelocity) };
	angVels.template storeWithOffset<OFFSET, 3, HK_IO_NATIVE_ALIGNED>((AngularVelocityDataType**) velsOut);
}

HK_FORCE_INLINE void hknpSolverVelocity::setVelocity(hkVector4Parameter linVel, hkVector4Parameter angVel)
{
#if !defined HK_PLATFORM_XBOX360
	// This is OK (even if we clobber angularVel[0]) because the next statement will set the angular velocity anyway.
	*vector0() = linVel;
	setAngularVelocity(angVel);
#else
	using namespace hknpSolverVelocity_Implementation;

	const hkQuadReal angVelShifted = __vsro( angVel.m_quad, *(const hkQuadReal*)&Masks::SetShiftMask );
	vector0()->m_quad = __vperm( linVel.m_quad, angVel.m_quad, *(const hkQuadReal*)&Masks::SetPermMask );
	__stvrx( angVelShifted, m_angularVelocity + 3, 0 );
#endif
}

HK_FORCE_INLINE void hknpSolverSumVelocity::setVelocity(hkVector4Parameter linVel, hkVector4Parameter angVel)
{
#if !defined HK_PLATFORM_XBOX360
	// This is OK (even if we clobber angularVel[0]) because the next statement will set the angular velocity anyway.
	*vector0() = linVel;
	setAngularVelocity(angVel);
#else
	using namespace hknpSolverVelocity_Implementation;

	const hkQuadReal angVelShifted = __vsro( angVel.m_quad, *(const hkQuadReal*)&Masks::SetShiftMask );
	vector0()->m_quad = __vperm( linVel.m_quad, angVel.m_quad, *(const hkQuadReal*)&Masks::SetPermMask );
	__stvrx( angVelShifted, m_angularVelocity + 3, 0 );
#endif
}

HK_FORCE_INLINE hkVector4* hknpSolverVelocity::linearVelocity()
{
	HK_ASSERT(0x84822003,(hkUlong(m_linearVelocity) & (HK_REAL_ALIGNMENT-1))==0);
	return (hkVector4*) m_linearVelocity;
}

HK_FORCE_INLINE const hkVector4* hknpSolverVelocity::linearVelocity() const
{
	HK_ASSERT(0x84822003,(hkUlong(m_linearVelocity) & (HK_REAL_ALIGNMENT-1))==0);
	return (const hkVector4*) m_linearVelocity;
}

HK_FORCE_INLINE hkReal* hknpSolverVelocity::angularVelocity()
{
	return m_angularVelocity;
}

HK_FORCE_INLINE const hkReal* hknpSolverVelocity::angularVelocity() const
{
	return m_angularVelocity;
}

HK_FORCE_INLINE hkVector4* hknpSolverVelocity::vector0()
{
	return linearVelocity();
}

HK_FORCE_INLINE hkVector4* hknpSolverVelocity::vector1()
{
	return vector0() + 1;
}

HK_FORCE_INLINE const hkVector4* hknpSolverVelocity::vector0() const
{
	return linearVelocity();
}

HK_FORCE_INLINE const hkVector4* hknpSolverVelocity::vector1() const
{
	return vector0() + 1;
}

HK_FORCE_INLINE hkVector4* hknpSolverSumVelocity::linearVelocity()
{
	HK_ASSERT(0x84822003,(hkUlong(m_linearVelocity) & (HK_REAL_ALIGNMENT-1))==0);
	return (hkVector4*) m_linearVelocity;
}

HK_FORCE_INLINE const hkVector4* hknpSolverSumVelocity::linearVelocity() const
{
	HK_ASSERT(0x84822003,(hkUlong(m_linearVelocity) & (HK_REAL_ALIGNMENT-1))==0);
	return (const hkVector4*) m_linearVelocity;
}

HK_FORCE_INLINE hkReal* hknpSolverSumVelocity::angularVelocity()
{
	return m_angularVelocity;
}

HK_FORCE_INLINE const hkReal* hknpSolverSumVelocity::angularVelocity() const
{
	return m_angularVelocity;
}

HK_FORCE_INLINE hkVector4* hknpSolverSumVelocity::vector0()
{
	return linearVelocity();
}

HK_FORCE_INLINE hkVector4* hknpSolverSumVelocity::vector1()
{
	return vector0() + 1;
}

HK_FORCE_INLINE const hkVector4* hknpSolverSumVelocity::vector0() const
{
	return linearVelocity();
}

HK_FORCE_INLINE const hkVector4* hknpSolverSumVelocity::vector1() const
{
	return vector0() + 1;
}

template<typename MXVECTOR> /* static */ HK_FORCE_INLINE void	hknpSolverVelocity::getVelocities(const hknpSolverVelocity* velocities, MXVECTOR& linVelsOut, MXVECTOR& angVelsOut)
{
	hknpSolverVelocity_Implementation::GetVelocitiesAction1<hknpSolverVelocity, MXVECTOR> action(velocities, linVelsOut, angVelsOut);
	hkMxUnroller<0, MXVECTOR::mxLength>::step(action);
}


template<typename MXVECTOR> /* static */ HK_FORCE_INLINE void	hknpSolverVelocity::setVelocities(const MXVECTOR& linVels, const MXVECTOR& angVels, hknpSolverVelocity* velsOut)
{
	hknpSolverVelocity_Implementation::SetVelocitiesAction1<hknpSolverVelocity, MXVECTOR> action(velsOut, linVels, angVels);
	hkMxUnroller<0, MXVECTOR::mxLength>::step(action);
}


template<typename MXVECTOR> /* static */ HK_FORCE_INLINE void	hknpSolverVelocity::getVelocities(const hknpSolverVelocity* velocities[], MXVECTOR& linVelsOut, MXVECTOR& angVelsOut)
{
	hknpSolverVelocity_Implementation::GetVelocitiesAction2<hknpSolverVelocity, MXVECTOR> action(velocities, linVelsOut, angVelsOut);
	hkMxUnroller<0, MXVECTOR::mxLength>::step(action);
}

template<typename MXVECTOR> /* static */ HK_FORCE_INLINE void	hknpSolverVelocity::getAll(const hknpSolverVelocity* velocities[], MXVECTOR& linVelsOut, MXVECTOR& angVelsOut, MXVECTOR& invIntertiasOut)
{
	hknpSolverVelocity_Implementation::GetAllAction2<hknpSolverVelocity, MXVECTOR> action(velocities, linVelsOut, angVelsOut, invIntertiasOut);
	hkMxUnroller<0, MXVECTOR::mxLength>::step(action);
}

template<typename MXVECTOR> /* static */ HK_FORCE_INLINE void	hknpSolverVelocity::setVelocities(const MXVECTOR& linVels, const MXVECTOR& angVels, hknpSolverVelocity** velsOut)
{
	hknpSolverVelocity_Implementation::SetVelocitiesAction2<hknpSolverVelocity, MXVECTOR> action(velsOut, linVels, angVels);
	hkMxUnroller<0, MXVECTOR::mxLength>::step(action);
}


HK_FORCE_INLINE	void hknpSolverVelocity::setInvInertias(const hkHalf* h)
{
	m_invInertiaTensor[0] = h[0]; m_invInertiaTensor[1] = h[1]; m_invInertiaTensor[2] = h[2]; m_invMassIntertia = h[3];
}


HK_FORCE_INLINE	void hknpSolverVelocity::getInvInertias(hkVector4& inertias) const
{
#if defined(HK_REAL_IS_DOUBLE)
	inertias.load<4, HK_IO_SIMD_ALIGNED>( &m_invInertiaTensor[0] );
#else
	hkHalf8Util::unpackSecond( *vector1(), inertias );
#endif
}


template<typename MXVECTOR>
/* static */ HK_FORCE_INLINE void	 hknpSolverVelocity::getInvInertias(const hknpSolverVelocity* velocities[], MXVECTOR& invIntertiasOut)
{
	hknpSolverVelocity_Implementation::GetInvInertiasAction1<hknpSolverVelocity, MXVECTOR> action(velocities, invIntertiasOut);
	hkMxUnroller<0, MXVECTOR::mxLength>::step(action);
}


template<typename MXVECTOR>
/* static */ HK_FORCE_INLINE void	 hknpSolverVelocity::setInvInertias(const MXVECTOR& invIntertias, hknpSolverVelocity** velsOut)
{
	enum { OFFSET = HK_OFFSET_OF(hknpSolverVelocity, m_invInertiaTensor) };
	invIntertias.template storePackWithOffset<OFFSET, 4, HK_IO_NATIVE_ALIGNED>((InvInertiasDataType**) velsOut);
}


HK_FORCE_INLINE void hknpSolverSumVelocity::init()
{
	vector0()->setZero();
	vector1()->setZero();
}


HK_FORCE_INLINE void hknpSolverVelocity::getLinearVelocity(hkVector4& velOut) const
{
	velOut = *linearVelocity();
}


HK_FORCE_INLINE void hknpSolverSumVelocity::getLinearVelocity(hkVector4& velOut) const
{
	velOut = *linearVelocity();
}


HK_FORCE_INLINE void hknpSolverVelocity::getAngularVelocity(hkVector4& velOut) const
{
#if !defined HK_PLATFORM_XBOX360
	velOut.setPermutation<hkVectorPermutation::WXYZ>(*vector1());
	velOut.setComponent<0>(vector0()->getComponent<3>());
#else
	using namespace hknpSolverVelocity_Implementation;
	velOut.m_quad = __vperm( vector0()->m_quad, vector1()->m_quad, *(const hkQuadReal*)&Masks::GetAngPermMask );
#endif
}


HK_FORCE_INLINE void hknpSolverSumVelocity::getAngularVelocity(hkVector4& velOut) const
{
#if !defined HK_PLATFORM_XBOX360
	velOut.setPermutation<hkVectorPermutation::WXYZ>(*vector1());
	velOut.setComponent<0>(vector0()->getComponent<3>());
#else
	using namespace hknpSolverVelocity_Implementation;
	velOut.m_quad = __vperm( vector0()->m_quad, vector1()->m_quad, *(const hkQuadReal*)&Masks::GetAngPermMask );
#endif
}


HK_FORCE_INLINE void hknpSolverSumVelocity::setAngularVelocity(hkVector4Parameter v)
{
	v.store<3, HK_IO_NATIVE_ALIGNED>(angularVelocity());
}




template<typename MXVECTOR> /* static */ HK_FORCE_INLINE void	hknpSolverSumVelocity::getVelocities(const hknpSolverSumVelocity* velocities, MXVECTOR& linVelsOut, MXVECTOR& angVelsOut)
{
	hknpSolverVelocity_Implementation::GetVelocitiesAction1<hknpSolverSumVelocity, MXVECTOR> action(velocities, linVelsOut, angVelsOut);
	hkMxUnroller<0, MXVECTOR::mxLength>::step(action);
}


template<typename MXVECTOR> /* static */ HK_FORCE_INLINE void	hknpSolverSumVelocity::setVelocities(const MXVECTOR& linVels, const MXVECTOR& angVels, hknpSolverSumVelocity* velsOut)
{
	hknpSolverVelocity_Implementation::SetVelocitiesAction1<hknpSolverSumVelocity, MXVECTOR> action(velsOut, linVels, angVels);
	hkMxUnroller<0, MXVECTOR::mxLength>::step(action);
}


template<typename MXVECTOR> /* static */ HK_FORCE_INLINE void	hknpSolverSumVelocity::getVelocities(const hknpSolverSumVelocity* velocities[], MXVECTOR& linVelsOut, MXVECTOR& angVelsOut)
{
	hknpSolverVelocity_Implementation::GetVelocitiesAction2<hknpSolverSumVelocity, MXVECTOR> action(velocities, linVelsOut, angVelsOut);
	hkMxUnroller<0, MXVECTOR::mxLength>::step(action);
}


template<typename MXVECTOR> /* static */ HK_FORCE_INLINE void	hknpSolverSumVelocity::setVelocities(const MXVECTOR& linVels, const MXVECTOR& angVels, hknpSolverSumVelocity** velsOut)
{
	hknpSolverVelocity_Implementation::SetVelocitiesAction2<hknpSolverSumVelocity, MXVECTOR> action(velsOut, linVels, angVels);
	hkMxUnroller<0, MXVECTOR::mxLength>::step(action);
}


HK_FORCE_INLINE const hkHalf& hknpSolverSumVelocity::getAngularSpeedLimit() const
{
	return m_angularSpeedLimit;
}


HK_FORCE_INLINE void hknpSolverSumVelocity::setAngularSpeedLimit(hkHalf value)
{
	m_angularSpeedLimit = value;
}


HK_FORCE_INLINE void hknpSolverSumVelocity::setAngularSpeedLimit(hkSimdRealParameter value)
{
	value.store<1>(&m_angularSpeedLimit);
}


HK_FORCE_INLINE const hknpMotionId& hknpSolverSumVelocity::getOriginalMotionId() const
{
	return m_originalMotionId;
}


HK_FORCE_INLINE void hknpSolverSumVelocity::setOriginalMotionId(hknpMotionId value)
{
	m_originalMotionId = value;
}


HK_FORCE_INLINE const hknpMotionPropertiesId& hknpSolverSumVelocity::getMotionPropertiesId() const
{
	return m_motionPropertiesId;
}


HK_FORCE_INLINE void hknpSolverSumVelocity::setMotionPropertiesId(hknpMotionPropertiesId value)
{
	m_motionPropertiesId = value;
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
