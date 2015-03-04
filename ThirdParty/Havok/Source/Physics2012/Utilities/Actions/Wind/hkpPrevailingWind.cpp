/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Physics2012/Utilities/Actions/Wind/hkpPrevailingWind.h>

// ////////////////////////////////////////////////////////////////////////
// OSCILLATING VECTOR
// ////////////////////////////////////////////////////////////////////////

hkpPrevailingWind::Oscillator::Oscillator( hkReal period, hkReal phase )
:	m_period( period ),
	m_accumulator( phase )
{
}

hkReal hkpPrevailingWind::Oscillator::getValue() const
{
	return hkMath::sin( 2.0f * HK_REAL_PI * m_accumulator );
}

void hkpPrevailingWind::Oscillator::update( hkReal delta )
{
	m_accumulator += delta / m_period;
	// need to limit the accumulator to the range [0,1].
	m_accumulator -= hkMath::floor( m_accumulator );
}

// ////////////////////////////////////////////////////////////////////////
// PREVAILING WIND
// ////////////////////////////////////////////////////////////////////////

hkpPrevailingWind::hkpPrevailingWind( const hkVector4& mid )
:	m_mid( mid ),
	m_current( mid )
{
}

void hkpPrevailingWind::getWindVector( const hkVector4 &pos, hkVector4& windOut ) const
{
	windOut = m_current;
}

void hkpPrevailingWind::addOscillation( const hkVector4& diff, hkReal period, hkReal power, hkReal phase )
{
	m_oscillators.pushBack( Triple( diff, Oscillator( period, phase ), power ) );
}

void hkpPrevailingWind::postSimulationCallback( hkpWorld* world )
{
	const hkReal delta = world->m_dynamicsStepInfo.m_stepInfo.m_deltaTime.val();
	m_current = m_mid;
	const int numOscillators = m_oscillators.getSize();
	for ( int i = 0; i < numOscillators; ++i )
	{
		m_oscillators[i].m_oscillator.update( delta );
		hkVector4 diff;
		{
			const hkReal val = m_oscillators[i].m_oscillator.getValue();
			const hkReal osc = val * hkMath::pow( hkMath::abs(val), m_oscillators[i].m_power - 1 );
			diff.setMul( hkSimdReal::fromFloat(osc), m_oscillators[i].m_diff );
		}
		m_current.add( diff );
	}
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
