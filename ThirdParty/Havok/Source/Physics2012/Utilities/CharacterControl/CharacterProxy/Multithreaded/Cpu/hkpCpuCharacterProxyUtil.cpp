/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Physics2012/Dynamics/World/hkpWorld.h>

#include <Physics2012/Utilities/CharacterControl/CharacterProxy/Multithreaded/Cpu/hkpCpuCharacterProxyUtil.h>
#include <Physics2012/Utilities/CharacterControl/CharacterProxy/hkpCharacterProxy.h>
#include <Physics2012/Collide/Query/CastUtil/hkpLinearCastInput.h>
#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Dynamics/Phantom/hkpSimpleShapePhantom.h>
#include <Physics2012/Dynamics/Phantom/hkpCachingShapePhantom.h>
#include <Physics2012/Collide/Query/Collector/PointCollector/hkpAllCdPointCollector.h>

void HK_CALL hkpCpuCharacterProxyUtil::linearCastPhantomCast( const hkpCharacterProxy* character, const hkpLinearCastInput& input, hkpAllCdPointCollector& castCollector, hkpAllCdPointCollector* startCollector )
{
	hkpShapePhantom* phantom = character->m_shapePhantom;

	//
	//	Setup the linear cast input
	//
	hkpLinearCastCollisionInput lcInput;
	{
		lcInput.set( *phantom->getWorld()->getCollisionInput() );

		lcInput.setPathAndTolerance( character->m_oldDisplacement, input.m_startPointTolerance );
		lcInput.m_maxExtraPenetration = input.m_maxExtraPenetration;
	}

	hkpPhantomType type = phantom->getType();
	const hkpCollidable* phantomCollidable = phantom->getCollidable();

	HK_ASSERT2(0xc034bea8 ,type == HK_PHANTOM_SIMPLE_SHAPE || type == HK_PHANTOM_CACHING_SHAPE, "phantom must be simple or caching shape phantom");

	//
	//	Do the cast.
	// 
	if( type == HK_PHANTOM_SIMPLE_SHAPE )
	{
		hkpSimpleShapePhantom* simplePhantom = static_cast<hkpSimpleShapePhantom*>(phantom);
		hkArray<hkpSimpleShapePhantom::CollisionDetail>& collisionDetatils = simplePhantom->getCollisionDetails();

		for ( int i = collisionDetatils.getSize() - 1; i >= 0; i-- )
		{	
			hkpSimpleShapePhantom::CollisionDetail& det = collisionDetatils[i];
			hkpShapeType typeB = det.m_collidable->getShape()->getType();
			hkpShapeType typeA = phantomCollidable->getShape()->getType();			
			hkpCollisionDispatcher::LinearCastFunc linearCastFunc = lcInput.m_dispatcher->getLinearCastFunc( typeA, typeB );
			linearCastFunc( *phantomCollidable, *det.m_collidable, lcInput, castCollector, startCollector );
		}
	}
	else // HK_PHANTOM_CACHING_SHAPE
	{
		hkpCachingShapePhantom* cachingPhantom =  (hkpCachingShapePhantom*)phantom;
		hkArray<hkpCachingShapePhantom::hkpCollisionDetail>& collisionDetatils = cachingPhantom->getCollisionDetails();

		for ( int i = collisionDetatils.getSize() - 1; i >= 0; i-- )
		{
			hkpCachingShapePhantom::hkpCollisionDetail& det = collisionDetatils[i];
			det.m_agent->linearCast( *phantomCollidable, *det.m_collidable, lcInput, castCollector, startCollector );
		}
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
