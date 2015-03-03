/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/Dynamics/World/ModifierManager/hknpModifierManager.h>
#include <Physics/Physics/Dynamics/Action/Manager/hknpActionManager.h>
#include <Physics/Internal/Dynamics/Solver/ConstraintAtom/hknpConstraintAtomSolver.h>
#include <Physics/Internal/Dynamics/World/Deactivation/hknpDeactivationManager.h>


HK_FORCE_INLINE const hkVector4& hknpWorld::getGravity() const
{
	return m_gravity;
}

HK_FORCE_INLINE void hknpWorld::setGravity( hkVector4Parameter gravity )
{
	m_gravity = gravity;
}

HK_FORCE_INLINE const hknpSolverInfo& hknpWorld::getSolverInfo() const
{
	return m_solverInfo;
}

HK_FORCE_INLINE hkUint32 hknpWorld::getNumFreeDynamicBodies()
{
	return hkMath::min2(
		m_bodyManager.getCapacity() - m_bodyManager.getNumAllocatedBodies(),
		m_motionManager.getCapacity() - m_motionManager.getNumAllocatedMotions() );
}

HK_FORCE_INLINE hkUint32 hknpWorld::getNumFreeStaticBodies()
{
	return m_bodyManager.getCapacity() - m_bodyManager.getNumAllocatedBodies();
}

HK_FORCE_INLINE hknpBodyId hknpWorld::createStaticBody( const hknpBodyCinfo& bodyCinfo, AdditionFlags additionFlags )
{
	checkNotInSimulation( SIMULATION_POST_COLLIDE );
	hknpBodyCinfo bodyCinfoCopy = bodyCinfo;
	bodyCinfoCopy.m_motionId = hknpMotionId::STATIC;
	return createBody( bodyCinfoCopy, additionFlags );
}

HK_FORCE_INLINE hknpBodyId hknpWorld::createDynamicBody(
	const hknpBodyCinfo& bodyCinfo, const hknpMotionCinfo& motionCinfo, AdditionFlags additionFlags )
{
	checkNotInSimulation( SIMULATION_POST_COLLIDE );
	hknpBodyCinfo bodyCinfoCopy = bodyCinfo;
	bodyCinfoCopy.m_motionId = createMotion( motionCinfo );
	return createBody( bodyCinfoCopy, additionFlags );
}

HK_FORCE_INLINE hkUint32 hknpWorld::getBodyCapacity() const
{
	return m_bodyManager.getCapacity();
}

HK_FORCE_INLINE hkUint32 hknpWorld::getStepCount()
{
	return m_solverInfo.m_stepSolveCount;
}

HK_FORCE_INLINE hkBool32 hknpWorld::isBodyValid( hknpBodyId bodyId ) const
{
	return ( bodyId.value() < m_bodyManager.getCapacity() ) &&
		m_bodyManager.m_bodies[bodyId.value()].isValid();
}

HK_FORCE_INLINE hkBool32 hknpWorld::isBodyAdded( hknpBodyId bodyId ) const
{
	return ( bodyId.value() < m_bodyManager.getCapacity() ) &&
		m_bodyManager.m_bodies[bodyId.value()].isAddedToWorld();
}

HK_FORCE_INLINE const hknpBody& hknpWorld::getBody( hknpBodyId bodyId ) const
{
	const hknpBody& body = m_bodyManager.getBody( bodyId );
	HK_ASSERT( 0x19b333d9, body.isValid() );
	return body;
}

HK_FORCE_INLINE const hknpBody& hknpWorld::getBodyUnchecked( hknpBodyId bodyId ) const
{
	const hknpBody& body = m_bodyManager.m_bodies[bodyId.value()];
	return body;
}

HK_FORCE_INLINE hknpBodyManager::BodyIterator hknpWorld::getBodyIterator() const
{
	return m_bodyManager.getBodyIterator();
}

HK_FORCE_INLINE hknpBody& hknpWorld::accessBody( hknpBodyId bodyId )
{
	hknpBody& body = m_bodyManager.accessBody( bodyId );
	HK_ASSERT( 0x19b333d9, body.isValid() );
	return body;
}

HK_FORCE_INLINE const hknpBody& hknpWorld::getSimulatedBody( hknpBodyId bodyId ) const
{
	const hknpBody& body = m_bodyManager.m_bodies[bodyId.value()];
	HK_ASSERT(0x19b333d9, body.isValid() && body.isAddedToWorld() );
	return body;
}

HK_FORCE_INLINE hkUint32 hknpWorld::getNumBodies() const
{
	return m_bodyManager.m_numAllocatedBodies;
}

#ifndef HK_PLATFORM_SPU

HK_FORCE_INLINE void hknpWorld::enableBodyFlags( hknpBodyId bodyId, hknpBody::Flags enabledFlags )
{
	HK_ASSERT( 0x78070a4b, isBodyValid(bodyId) );
	HK_ASSERT2( 0xf034dfdf, 0 == (enabledFlags.anyIsSet(hknpBody::INTERNAL_FLAGS_MASK)), "Cannot enable these body flags");
	enabledFlags.clear( hknpBody::INTERNAL_FLAGS_MASK );
	hknpBody& body = m_bodyManager.accessBody(bodyId);
	if( !body.m_flags.allAreSet(enabledFlags.get()) )
	{
		body.m_flags.orWith( enabledFlags.get() );
		rebuildBodyCollisionCaches( bodyId );
		m_signals.m_bodyChanged.fire( this, bodyId );
	}
}

HK_FORCE_INLINE void hknpWorld::disableBodyFlags( hknpBodyId bodyId, hknpBody::Flags disabledFlags )
{
	HK_ASSERT( 0x78070a4b, isBodyValid(bodyId) );
	HK_ASSERT2( 0xf034dfdf, 0 == (disabledFlags.anyIsSet(hknpBody::INTERNAL_FLAGS_MASK)), "Cannot disable these body flags");
	disabledFlags.clear( hknpBody::INTERNAL_FLAGS_MASK );
	hknpBody& body = m_bodyManager.accessBody(bodyId);
	if( body.m_flags.anyIsSet(disabledFlags.get()) )
	{
		body.m_flags.clear( disabledFlags.get() );
		rebuildBodyCollisionCaches( bodyId );
		m_signals.m_bodyChanged.fire( this, bodyId );
	}
}

HK_FORCE_INLINE void hknpWorld::enableBodySpuFlags( hknpBodyId bodyId, hknpBody::SpuFlags enabledFlags )
{
	HK_ASSERT( 0x78070a4b, isBodyValid(bodyId) );
	hknpBody& body = m_bodyManager.accessBody(bodyId);
	if( !body.m_spuFlags.allAreSet(enabledFlags.get()) )
	{
		body.m_spuFlags.orWith( enabledFlags.get() );
		rebuildBodyCollisionCaches( bodyId );
		m_signals.m_bodyChanged.fire( this, bodyId );
	}
}

HK_FORCE_INLINE void hknpWorld::disableBodySpuFlags( hknpBodyId bodyId, hknpBody::SpuFlags disabledFlags )
{
	HK_ASSERT( 0x78070a4b, isBodyValid(bodyId) );
	hknpBody& body = m_bodyManager.accessBody(bodyId);
	if( body.m_spuFlags.anyIsSet( disabledFlags.get()) )
	{
		body.m_spuFlags.clear( disabledFlags.get() );
		rebuildBodyCollisionCaches( bodyId );
		m_signals.m_bodyChanged.fire( this, bodyId );
	}
}

#endif

HK_FORCE_INLINE hknpModifierManager* hknpWorld::getModifierManager()
{
	return m_modifierManager;
}

HK_FORCE_INLINE hknpEventDispatcher* hknpWorld::getEventDispatcher() const
{
	return m_eventDispatcher;
}

HK_FORCE_INLINE void hknpWorld::setCollisionFilter( hknpCollisionFilter* collisionFilter )
{
	m_modifierManager->setCollisionFilter( collisionFilter );
}

HK_FORCE_INLINE hknpCollisionFilter* hknpWorld::getCollisionFilter() const
{
	return m_modifierManager->getCollisionFilter();
}

HK_FORCE_INLINE void hknpWorld::setShapeTagCodec( hknpShapeTagCodec* shapeTagCodec )
{
	if( shapeTagCodec == HK_NULL )
	{
		m_shapeTagCodec = &m_nullShapeTagCodec;
	}
	else
	{
		m_shapeTagCodec = shapeTagCodec;
	}
}

HK_FORCE_INLINE const hknpShapeTagCodec* hknpWorld::getShapeTagCodec() const
{
	return m_shapeTagCodec;
}

HK_FORCE_INLINE hkBool32 hknpWorld::isDeactivationEnabled() const
{
	return m_deactivationEnabled;
}

#if !defined(HK_PLATFORM_SPU)

HK_FORCE_INLINE void hknpWorld::setBodyDeactivationEnabled( hknpBodyId bodyId, bool enableDeactivation )
{
	if (isDeactivationEnabled())
	{
		m_deactivationManager->setBodyDeactivation(bodyId, enableDeactivation);
	}
}

#endif

HK_FORCE_INLINE const hkArray<hknpBodyId>& hknpWorld::getActiveBodies()
{
	const hkArray<hknpBodyId>& activeBodies = m_bodyManager.getActiveBodies();
	return activeBodies;
}

HK_FORCE_INLINE const hknpMotion& hknpWorld::getMotion( hknpMotionId motionId ) const
{
	HK_ASSERT( 0x34ec543e, m_motionManager.isMotionValid(motionId) );
	return m_motionManager.m_motions[motionId.value()];
}

HK_FORCE_INLINE const hkTransform& hknpWorld::getBodyTransform( hknpBodyId bodyId ) const
{
	return getBody(bodyId).getTransform();
}

HK_FORCE_INLINE void hknpWorld::getBodyCinfo( hknpBodyId bodyId, hknpBodyCinfo& bodyCinfoOut ) const
{
	return m_bodyManager.getBodyCinfo( bodyId, bodyCinfoOut );
}

HK_FORCE_INLINE void hknpWorld::getMotionCinfo( hknpMotionId motionId, hknpMotionCinfo& motionCinfoOut ) const
{
	return m_motionManager.getMotionCinfo( motionId, motionCinfoOut );
}

HK_FORCE_INLINE hknpMotion& hknpWorld::accessMotion( hknpMotionId motionId )
{
	checkNotInSimulation( SIMULATION_PRE_COLLIDE | SIMULATION_POST_COLLIDE | SIMULATION_PRE_SOLVE | SIMULATION_POST_SOLVE );
	HK_ASSERT( 0x498380c8, m_motionManager.isMotionValid(motionId) );
	hknpMotion& motion = m_motionManager.m_motions[motionId.value()];
	// If the motion is inactive but already added to the world, we can't access it.
	HK_ASSERT( 0xaf1fe122, !motion.m_firstAttachedBodyId.isValid() || !getBody(motion.m_firstAttachedBodyId).isAddedToWorld() || !getBody(motion.m_firstAttachedBodyId).isInactive() );
	return motion;
}

HK_FORCE_INLINE hknpMotion& hknpWorld::accessMotionUnchecked( hknpMotionId motionId )
{
	HK_ASSERT( 0x498380c8, m_motionManager.isMotionValid(motionId) );
	return m_motionManager.m_motions[motionId.value()];
}

HK_FORCE_INLINE const hknpMaterialLibrary* hknpWorld::getMaterialLibrary() const
{
	return m_materialLibrary;
}

HK_FORCE_INLINE hknpMaterialLibrary* hknpWorld::accessMaterialLibrary()
{
	checkNotInSimulation(SIMULATION_POST_COLLIDE);
	return m_materialLibrary;
}

HK_FORCE_INLINE const hknpBodyQualityLibrary* hknpWorld::getBodyQualityLibrary() const
{
	return m_qualityLibrary;
}

HK_FORCE_INLINE hknpBodyQualityLibrary* hknpWorld::accessBodyQualityLibrary()
{
	checkNotInSimulation(SIMULATION_POST_COLLIDE);
	return m_qualityLibrary;
}

HK_FORCE_INLINE const hknpMotionPropertiesLibrary* hknpWorld::getMotionPropertiesLibrary() const
{
	return m_motionPropertiesLibrary;
}

HK_FORCE_INLINE hknpMotionPropertiesLibrary* hknpWorld::accessMotionPropertiesLibrary()
{
	checkNotInSimulation(SIMULATION_POST_COLLIDE);
	return m_motionPropertiesLibrary;
}

#if !defined ( HK_PLATFORM_SPU )

HK_FORCE_INLINE void hknpWorld::findBodyConstraints( hknpBodyId bodyId, hkArray<const hknpConstraint*>& constraintsOut ) const
{
	m_constraintAtomSolver->findConstraintsUsingBody( bodyId, constraintsOut );
}

HK_FORCE_INLINE hkBool32 hknpWorld::isConstraintAdded( const hknpConstraint* constraint ) const
{
	return m_constraintAtomSolver->isConstraintAdded( constraint );
}

#endif

HK_FORCE_INLINE void hknpWorld::checkNotInSimulation( int allowedStates ) const
{
	HK_ASSERT( 0xf03dfd40, m_simulationStage == SIMULATION_DONE || m_simulationStage == SIMULATION_POST_SOLVE || (m_simulationStage&allowedStates) != 0 );
}

HK_FORCE_INLINE void hknpWorld::setBodyStatic(	hknpBodyId bodyId, RebuildCachesMode cacheBehavior )
{
	setBodyMotion( bodyId, hknpMotionId::STATIC, cacheBehavior );
}

HK_FORCE_INLINE void hknpWorld::setBodyKeyframed( hknpBodyId bodyId, RebuildCachesMode cacheBehavior )
{
	setBodyMass( bodyId, 0.0f );
}


template< typename T >
HK_FORCE_INLINE void hknpWorld::setBodyProperty( hknpBodyId bodyId, hknpPropertyKey key, const T& value )
{
	m_bodyManager.setProperty<T>( bodyId, key, value );
}

template< typename T >
HK_FORCE_INLINE T* hknpWorld::getBodyProperty( hknpBodyId bodyId, hknpPropertyKey key ) const
{
	return m_bodyManager.getProperty<T>( bodyId, key );
}

HK_FORCE_INLINE void hknpWorld::clearBodyProperty( hknpBodyId bodyId, hknpPropertyKey key )
{
	m_bodyManager.clearProperty( bodyId, key );
}

HK_FORCE_INLINE void hknpWorld::clearPropertyFromAllBodies( hknpPropertyKey key )
{
	m_bodyManager.clearPropertyFromAllBodies( key );
}

HK_FORCE_INLINE void hknpWorld::clearAllPropertiesFromBody( hknpBodyId bodyId )
{
	m_bodyManager.clearAllPropertiesFromBody( bodyId );
}

HK_FORCE_INLINE void hknpWorld::clearAllPropertiesFromAllBodies()
{
	m_bodyManager.clearAllPropertiesFromAllBodies();
}


HK_FORCE_INLINE void hknpWorld::setConsistencyChecksEnabled( bool areEnabled )
{
#if defined(HK_DEBUG)
	m_consistencyChecksEnabled = areEnabled;
#endif
}

HK_FORCE_INLINE hkBool32 hknpWorld::areConsistencyChecksEnabled() const
{
#if defined(HK_DEBUG)
	return m_consistencyChecksEnabled;
#else
	return false;	// allows the compiler to optimize code away
#endif
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
