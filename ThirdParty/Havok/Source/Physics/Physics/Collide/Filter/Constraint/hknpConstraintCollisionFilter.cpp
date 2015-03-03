/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Collide/Filter/Constraint/hknpConstraintCollisionFilter.h>


hknpConstraintCollisionFilter::hknpConstraintCollisionFilter( const hknpCollisionFilter* childFilter )
:	hknpPairCollisionFilter( childFilter )
{
	m_type = hknpCollisionFilter::CONSTRAINT_FILTER;
	m_subscribedWorld = HK_NULL;
}

hknpConstraintCollisionFilter::hknpConstraintCollisionFilter( hkFinishLoadedObjectFlag flag )
:	hknpPairCollisionFilter( flag )
{
}

hknpConstraintCollisionFilter::~hknpConstraintCollisionFilter()
{
	unsubscribeFromWorld();
}

void hknpConstraintCollisionFilter::updateFromWorld( hknpWorld* world )
{
	hknpPairCollisionFilter::clearAll();

	const int numConstraints = world->m_constraintAtomSolver->getNumConstraints();
	for ( int i=0; i<numConstraints; i++ )
	{
		hknpConstraint* constraint = world->m_constraintAtomSolver->getConstraints()[i];
		disableCollisionsBetween( world, constraint->m_bodyIdA, constraint->m_bodyIdB );
	}
}

void hknpConstraintCollisionFilter::subscribeToWorld( hknpWorld* world )
{
	unsubscribeFromWorld();
	world->m_signals.m_constraintAdded.subscribe( this, &hknpConstraintCollisionFilter::onConstraintAddedSignal, "hknpConstraintCollisionFilter" );
	world->m_signals.m_constraintRemoved.subscribe( this, &hknpConstraintCollisionFilter::onConstraintRemovedSignal, "hknpConstraintCollisionFilter" );
	m_subscribedWorld = world;
}

void hknpConstraintCollisionFilter::unsubscribeFromWorld()
{
	if( m_subscribedWorld )
	{
		m_subscribedWorld->m_signals.m_constraintAdded.unsubscribeAll( this );
		m_subscribedWorld->m_signals.m_constraintRemoved.unsubscribeAll( this );
		m_subscribedWorld = HK_NULL;
	}
}

void hknpConstraintCollisionFilter::onConstraintAddedSignal( hknpWorld* world, hknpConstraint* constraint )
{
	disableCollisionsBetween( world, constraint->m_bodyIdA, constraint->m_bodyIdB );
}

void hknpConstraintCollisionFilter::onConstraintRemovedSignal( hknpWorld* world, hknpConstraint* constraint )
{
	enableCollisionsBetween( world, constraint->m_bodyIdA, constraint->m_bodyIdB );
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
