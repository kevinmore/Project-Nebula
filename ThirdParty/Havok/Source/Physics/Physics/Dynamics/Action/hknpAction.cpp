/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Dynamics/Action/hknpAction.h>

#include <Physics/Physics/Dynamics/World/Deactivation/hknpCollisionPair.h>


void hknpUnaryAction::getBodies( hkArray<hknpBodyId>* HK_RESTRICT bodiesOut ) const
{
	bodiesOut->pushBackUnchecked( m_body );
}

void hknpBinaryAction::getBodies( hkArray<hknpBodyId>* HK_RESTRICT bodiesOut ) const
{
	bodiesOut->pushBackUnchecked( m_bodyA );
	bodiesOut->pushBackUnchecked( m_bodyB );
}

void hknpBinaryAction::initialize( hknpBodyId idA, hknpBodyId idB, hkUlong userData )
{
	m_bodyA = idA;
	m_bodyB = idB;
	m_userData = userData;
}

void hknpAction::addLinkUnchecked( const hknpMotion* motionA, const hknpMotion* motionB, hknpCdPairWriter* HK_RESTRICT pairWriter )
{
	HK_ASSERT( 0xf054fcde, motionA->isDynamic() && motionB->isDynamic() );
	hknpCollisionPair* HK_RESTRICT activePair = pairWriter->reserve( sizeof( hknpCollisionPair ) ) ;
	activePair->m_cell[0] = motionA->m_cellIndex;
	activePair->m_cell[1] = motionB->m_cellIndex;
	activePair->m_id[0] = motionA->m_solverId;
	activePair->m_id[1] = motionB->m_solverId;
	pairWriter->advance( sizeof( hknpCollisionPair ) );
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
