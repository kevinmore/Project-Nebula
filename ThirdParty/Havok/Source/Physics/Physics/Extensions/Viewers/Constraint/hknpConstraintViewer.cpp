/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Extensions/Viewers/Constraint/hknpConstraintViewer.h>

#include <Physics/Internal/Dynamics/Solver/ConstraintAtom/hknpConstraintAtomSolver.h>
#include <Physics/Internal/Dynamics/Solver/ConstraintAtom/hknpConstraintAtomSolverSetup.h>
#include <Physics/ConstraintSolver/Jacobian/hkpJacobianSchema.h>
#include <Physics/Constraint/Visualize/DrawDispatcher/hkpDrawDispatcher.h>

#include <Common/Visualize/hkProcessFactory.h>


int hknpConstraintViewer::s_tag = 0;
hkReal hknpConstraintViewer::m_scale = 0.25f;

void HK_CALL hknpConstraintViewer::registerViewer(hkProcessFactory& factory)
{
	s_tag = factory.registerProcess( getName(), create );
}

hkProcess* HK_CALL hknpConstraintViewer::create(const hkArray<hkProcessContext*>& contexts)
{
	return new hknpConstraintViewer( contexts );
}


hknpConstraintViewer::hknpConstraintViewer(const hkArray<hkProcessContext*>& contexts)
	: hknpViewer( contexts )
{
}

hknpConstraintViewer::~hknpConstraintViewer()
{
	if ( m_context )
	{
		for ( int i=0; i < m_context->getNumWorlds(); ++i )
		{
			worldRemovedCallback( m_context->getWorld(i) );
		}
	}
}

void hknpConstraintViewer::worldAddedCallback( hknpWorld* world )
{
	HK_SUBSCRIBE_TO_SIGNAL( world->m_signals.m_immediateConstraintAdded, this, hknpConstraintViewer );
}

void hknpConstraintViewer::worldRemovedCallback( hknpWorld* world )
{
	world->m_signals.m_immediateConstraintAdded.unsubscribeAll(this);
}

void hknpConstraintViewer::onImmediateConstraintAddedSignal( hknpWorld* world, const hknpConstraint* constraint )
{
	const hknpBody& bodyA = world->getBody(constraint->m_bodyIdA);
	const hknpBody& bodyB = world->getBody(constraint->m_bodyIdB);

	hkpDispatchDraw( constraint->m_data, bodyA.getTransform(), bodyB.getTransform(), m_displayHandler, bodyA.m_motionId.value(), s_tag, m_scale );
}

void hknpConstraintViewer::step( hkReal deltaTime )
{
	if( !m_context )
	{
		return;
	}

	HK_TIMER_BEGIN( "ConstraintViewer", this );

	for( int wi=0; wi<m_context->getNumWorlds(); wi++ )
	{
		hknpWorld* world = m_context->getWorld(wi);

		// find all persistent constraints and draw them
		for( int i = 0; i < world->m_constraintAtomSolver->getNumConstraints(); ++i )
		{
			const hknpConstraint* constraint = world->m_constraintAtomSolver->getConstraints()[i];

			const hknpBody& bodyA = world->getBody(constraint->m_bodyIdA);
			const hknpBody& bodyB = world->getBody(constraint->m_bodyIdB);

			hkpDispatchDraw( constraint->m_data, bodyA.getTransform(), bodyB.getTransform(), m_displayHandler, bodyA.m_motionId.value(), s_tag, m_scale );
		}
	}

	HK_TIMER_END();
}

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
