/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>

#include <Common/Base/Monitor/hkMonitorStream.h>

#include <Common/Base/Algorithm/Sort/hkSort.h>

#include <Physics2012/Dynamics/Phantom/hkpPhantomType.h>
#include <Physics2012/Dynamics/Phantom/hkpAabbPhantom.h>
#include <Common/Base/Types/Geometry/Aabb/hkAabb.h>
#include <Physics2012/Dynamics/World/hkpWorld.h>
#include <Physics2012/Dynamics/World/hkpWorldObject.h>
#include <Physics2012/Collide/Filter/hkpCollisionFilter.h>

#include <Physics2012/Collide/Query/CastUtil/hkpWorldRayCastInput.h>
#include <Physics2012/Collide/Query/CastUtil/hkpWorldRayCastOutput.h>
#include <Physics2012/Collide/Shape/Query/hkpRayHitCollector.h>

#include <Physics2012/Collide/Query/CastUtil/hkpLinearCastInput.h>
#include <Physics2012/Collide/Agent/Query/hkpLinearCastCollisionInput.h>
#include <Physics2012/Collide/Dispatch/hkpCollisionDispatcher.h>
#include <Physics2012/Collide/Shape/Query/hkpShapeRayCastInput.h>

hkpAabbPhantom::hkpAabbPhantom( const hkAabb& aabb, hkUint32 collisionFilterInfo )
:	hkpPhantom( HK_NULL ),
	m_orderDirty( false )
{
	m_aabb = aabb;
	HK_ASSERT(0x16b2d783,  m_aabb.isValid());

	m_collidable.setCollisionFilterInfo( collisionFilterInfo );
}

hkpAabbPhantom::~hkpAabbPhantom()
{
}

hkpPhantomType hkpAabbPhantom::getType() const
{
	return HK_PHANTOM_AABB;
}

// hkpPhantom clone functionality
hkpPhantom* hkpAabbPhantom::clone() const
{
	const hkpCollidable* c = getCollidable();
	hkpAabbPhantom* ap = new hkpAabbPhantom( m_aabb, c->getCollisionFilterInfo() );

	// stuff that isn 't in the ctor (the array of listeners (not ref counted etc, assume clone wants the same list to start with)
	ap->m_overlapListeners = m_overlapListeners;
	ap->m_phantomListeners = m_phantomListeners;

	ap->copyProperties( this );
	return ap;
}

void hkpAabbPhantom::calcAabb( hkAabb& aabb )
{
	aabb = m_aabb;
}

void hkpAabbPhantom::setAabb(const hkAabb& newAabb)
{
	HK_ACCESS_CHECK_WITH_PARENT( m_world, HK_ACCESS_RO, this, HK_ACCESS_RW );
	m_aabb = newAabb;
	HK_ASSERT(0x270ca8e7,  m_aabb.isValid());

	updateBroadPhase( m_aabb );
}


void hkpAabbPhantom::castRay( const hkpWorldRayCastInput& input, hkpRayHitCollector& collector ) const
{
	HK_ACCESS_CHECK_WITH_PARENT( m_world, HK_ACCESS_RO, this, HK_ACCESS_RO );
	HK_ASSERT2(0x5fc4e16f, collector.m_earlyOutHitFraction == 1.0f, "Your collector is not reset");
	HK_ASSERT2(0x16656fd2, input.m_from.isOk<3>(),	"Input 'from' vector is invalid.");
	HK_ASSERT2(0x1d9b6f86, input.m_to.isOk<3>(),		"Input 'to' vector is invalid.");

#ifdef HK_DEBUG
	//
	//	Check whether our ray is inside the AABB
	//
	{
		hkAabb rayAabb;
		rayAabb.m_min.setMin( input.m_from, input.m_to );
		rayAabb.m_max.setMax( input.m_from, input.m_to );
		HK_ASSERT2(0x7a3770bc,  m_aabb.contains(rayAabb ), "The AABB of the hkpAabbPhantom does not include the ray. Did you forget to call setAabb");
	}
#endif

	HK_TIMER_BEGIN("rcPhantom", HK_NULL);

	hkpShapeRayCastInput sinput;
	sinput.m_filterInfo = input.m_filterInfo;

	if (input.m_enableShapeCollectionFilter )
	{
		sinput.m_rayShapeCollectionFilter = m_world->getCollisionFilter();
	}
	else
	{
		sinput.m_rayShapeCollectionFilter = HK_NULL;
	}

	sinput.m_userData =  input.m_userData;

	const hkpCollidable* const* col = m_overlappingCollidables.begin();

	for (int i = m_overlappingCollidables.getSize()-1; i >=0 ; i--)
	{

		const hkpShape* shape = (*col)->getShape();
		if ( shape != HK_NULL )
		{
			const hkTransform& trans = (*col)->getTransform();
			sinput.m_from._setTransformedInversePos( trans, input.m_from );
			sinput.m_to.  _setTransformedInversePos( trans, input.m_to );
			sinput.m_collidable = *col;
			const hkpCdBody& body = **col;

			shape->castRayWithCollector( sinput, body, collector );
		}
		col++;
	}
	HK_TIMER_END();
}

void hkpAabbPhantom::castRay( const hkpWorldRayCastInput& input, hkpWorldRayCastOutput& output ) const
{
	HK_ACCESS_CHECK_WITH_PARENT( m_world, HK_ACCESS_RO, this, HK_ACCESS_RO );
	HK_ASSERT2(0x1344cc46, output.m_rootCollidable == HK_NULL, "Your output has not been reset");
	HK_ASSERT2(0x16656fd2, input.m_from.isOk<3>(),	"Input 'from' vector is invalid.");
	HK_ASSERT2(0x1d9b6f86, input.m_to.isOk<3>(),		"Input 'to' vector is invalid.");

#ifdef HK_DEBUG
	//
	//	Check whether our ray is inside the AABB
	//
	{
		hkAabb rayAabb;
		rayAabb.m_min.setMin( input.m_from, input.m_to );
		rayAabb.m_max.setMax( input.m_from, input.m_to );
		HK_ASSERT2(0x26c49b67,  m_aabb.contains(rayAabb ), "The AABB of the hkpAabbPhantom does not include the ray. Did you forget to call setAabb");
	}
#endif
	HK_TIMER_BEGIN("rcPhantom", HK_NULL);

	hkpShapeRayCastOutput rayResults;

	hkpShapeRayCastInput sinput;
	sinput.m_filterInfo = input.m_filterInfo;

	if (input.m_enableShapeCollectionFilter )
	{
		sinput.m_rayShapeCollectionFilter = m_world->getCollisionFilter();
	}
	else
	{
		sinput.m_rayShapeCollectionFilter = HK_NULL;
	}

	sinput.m_userData =  input.m_userData;
	
	const hkpCollidable* const * col = m_overlappingCollidables.begin();

	for (int i = m_overlappingCollidables.getSize()-1; i >=0 ; i--)
	{
		const hkpShape* shape = (*col)->getShape();
		if ( shape != HK_NULL )
		{
			const hkTransform& trans = (*col)->getTransform();
			sinput.m_from._setTransformedInversePos( trans, input.m_from );
			sinput.m_to.  _setTransformedInversePos( trans, input.m_to );
			sinput.m_collidable = *col;
			
			hkBool hit = shape->castRay( sinput, output);
			if (hit)
			{
				output.m_rootCollidable = *col;
			}
		}
		col++;
	}

	if ( output.m_rootCollidable != HK_NULL )
	{
		const hkTransform& trans = output.m_rootCollidable->getTransform();
		output.m_normal.setRotatedDir( trans.getRotation(), output.m_normal);
	}
	HK_TIMER_END();
}

void hkpAabbPhantom::linearCast(const hkpCollidable* const toBeCast, const hkpLinearCastInput& input, hkpCdPointCollector& castCollector, hkpCdPointCollector* startCollector ) const
{
	HK_ACCESS_CHECK_WITH_PARENT( m_world, HK_ACCESS_RO, this, HK_ACCESS_RO );
	HK_ASSERT2(0xDBC00004, toBeCast, "hkpCollidable toBeCast is null.");
	HK_TIME_CODE_BLOCK("hkpAabbPhantom::linearCast", HK_NULL);

	hkVector4 path;
	path.setSub(input.m_to, toBeCast->getTransform().getTranslation());

	// Setup the linear cast input
	hkpLinearCastCollisionInput lccInput;
	{
		lccInput.set( *m_world->getCollisionInput() );
		lccInput.setPathAndTolerance( path, input.m_startPointTolerance );
		lccInput.m_maxExtraPenetration = input.m_maxExtraPenetration;
	}

	const hkpCollisionDispatcher* dispatcher	= m_world->getCollisionDispatcher();
	const hkpCollisionFilter*	 filter		= m_world->getCollisionFilter();
	HK_ASSERT2(0xDBC00001, dispatcher , "World's collision dispatcher is null.");
	HK_ASSERT2(0xDBC00002, filter,		"World's collision filter is null");

	const int numCollidables = m_overlappingCollidables.getSize();
	for ( int i = 0; i<numCollidables; i++ )
	{
		const hkpCollidable* target = m_overlappingCollidables[i];
		HK_ASSERT2(0xDBC00003, target, "An overlapping collidable is null.");

		const bool collisionEnabled = filter->isCollisionEnabled(*toBeCast, *target);
		const bool differentCollidable = (target != toBeCast);

		if (target->getShape() && collisionEnabled && differentCollidable)
		{
			const hkpShapeType typeA = toBeCast->getShape()->getType();
			const hkpShapeType typeB = target->getShape()->getType();
			hkpCollisionDispatcher::LinearCastFunc linearCastFunc = dispatcher->getLinearCastFunc(typeA, typeB);
			linearCastFunc(*toBeCast, *target, lccInput, castCollector, startCollector);
		}
	}
}

hkBool hkpAabbPhantom::isOverlappingCollidableAdded( const hkpCollidable* collidable )
{
	HK_ACCESS_CHECK_WITH_PARENT( m_world, HK_ACCESS_IGNORE, this, HK_ACCESS_RO );
	for( int i = 0; i<m_overlappingCollidables.getSize(); i++ )
	{
		if( m_overlappingCollidables[i] == collidable )
		{
			return true;
		}
	}
	return false;
}

void hkpAabbPhantom::addOverlappingCollidable( hkpCollidable* collidable )
{
	// note: the filtering happens before this function is called

	HK_ACCESS_CHECK_WITH_PARENT( m_world, HK_ACCESS_RO, this, HK_ACCESS_RW );

	hkpCollidableAccept accept = fireCollidableAdded( collidable );

	if ( accept == HK_COLLIDABLE_ACCEPT  )
	{
		 m_overlappingCollidables.pushBack( collidable );
		 m_orderDirty = true;
	}
}

void hkpAabbPhantom::removeOverlappingCollidable( hkpCollidable* collidable )
{
	HK_ACCESS_CHECK_WITH_PARENT( m_world, HK_ACCESS_RO, this, HK_ACCESS_RW );

	// Get the index of the collidable and remove it from the list (with index validity check)
	int index = m_overlappingCollidables.indexOf( collidable );

	fireCollidableRemoved( collidable, index >= 0 );

	if( index >= 0 )
	{
		m_overlappingCollidables.removeAt( index );
		m_orderDirty = true;
	}

}

void hkpAabbPhantom::deallocateInternalArrays()
{
	HK_ACCESS_CHECK_WITH_PARENT( m_world, HK_ACCESS_RO, this, HK_ACCESS_RW );

	// Need to deallocate any arrays that are 0 size
	// else warn user that they should call the in place destructor

	// Overlap Listeners
	if (m_overlappingCollidables.getSize() == 0)
	{
		m_overlappingCollidables.clearAndDeallocate();
	}
	else
	{
		HK_WARN(0x234f224e, "Phantom at address " << this << " has non-zero m_overlappingCollidables array.\nPlease call in-place destructor to deallocate.\n");
	}

	hkpPhantom::deallocateInternalArrays();
}

class hkpAabbPhantom::OrderByUid
{
	public:
		HK_FORCE_INLINE hkBool operator() ( const hkpCollidable* a, const hkpCollidable* b )
		{
			return a->getBroadPhaseHandle()->m_id < b->getBroadPhaseHandle()->m_id;
		}
};

void hkpAabbPhantom::ensureDeterministicOrder()
{
	// Ensure determinism by sorting on broadphase id.
	if ( m_orderDirty )
	{
		hkSort( m_overlappingCollidables.begin(), m_overlappingCollidables.getSize(), OrderByUid() );
		m_orderDirty = false;
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
