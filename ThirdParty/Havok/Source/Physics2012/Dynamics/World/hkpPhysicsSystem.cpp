/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Physics2012/Dynamics/World/hkpPhysicsSystem.h>
#include <Physics2012/Dynamics/Phantom/hkpPhantom.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Dynamics/Action/hkpAction.h>
#include <Common/Base/Types/Geometry/Aabb/hkAabb.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>

#include <Physics/Constraint/Data/hkpConstraintData.h>
#include <Physics2012/Dynamics/Constraint/hkpConstraintInstance.h>
#include <Physics/Constraint/Data/BallAndSocket/hkpBallAndSocketConstraintData.h>
#include <Physics2012/Dynamics/Constraint/Breakable/hkpBreakableConstraintData.h>
#include <Physics/Constraint/Data/Hinge/hkpHingeConstraintData.h>
#include <Physics/Constraint/Data/LimitedHinge/hkpLimitedHingeConstraintData.h>
#include <Physics2012/Dynamics/Constraint/Malleable/hkpMalleableConstraintData.h>
#include <Physics/Constraint/Data/PointToPath/hkpPointToPathConstraintData.h>
#include <Physics/Constraint/Data/PointToPlane/hkpPointToPlaneConstraintData.h>
#include <Physics/Constraint/Data/Prismatic/hkpPrismaticConstraintData.h>
#include <Physics/Constraint/Data/Ragdoll/hkpRagdollConstraintData.h>
#include <Physics/Constraint/Data/StiffSpring/hkpStiffSpringConstraintData.h>
#include <Physics/Constraint/Data/Wheel/hkpWheelConstraintData.h>
#include <Physics2012/Dynamics/Constraint/Util/hkpConstraintDataCloningUtil.h>
#include <Physics2012/Dynamics/Constraint/Atom/hkpSimpleContactConstraintAtom.h>
#include <Physics2012/Dynamics/Phantom/hkpSimpleShapePhantom.h>
#include <Physics2012/Dynamics/Phantom/hkpCachingShapePhantom.h>
#include <Physics2012/Dynamics/Phantom/hkpAabbPhantom.h>

hkpPhysicsSystem::hkpPhysicsSystem()
: m_name(HK_NULL), m_userData(HK_NULL), m_active(true)
{
}

void hkpPhysicsSystem::removeAll()
{
	int nrb = m_rigidBodies.getSize();
	for (int r=0; r < nrb; ++r)
	{
		if(m_rigidBodies[r])
		{
			m_rigidBodies[r]->removeReference();
		}
	}
	m_rigidBodies.clear();

	int np = m_phantoms.getSize();
	for (int p=0; p < np; ++p)
	{
		if (m_phantoms[p])
		{
			m_phantoms[p]->removeReference();
		}
	}
	m_phantoms.clear();

	int nc = m_constraints.getSize();
	for (int c=0; c < nc; ++c)
	{
		if (m_constraints[c]) 
		{
			m_constraints[c]->removeReference();
		}
	}
	m_constraints.clear();

	int na = m_actions.getSize();
	for (int a=0; a < na; ++a)
	{
		if (m_actions[a]) 
		{
			m_actions[a]->removeReference();
		}
	}
	m_actions.clear();
}

hkpPhysicsSystem::~hkpPhysicsSystem()
{
	removeAll();
}

void hkpPhysicsSystem::copy(const hkpPhysicsSystem& toCopy)
{
	m_rigidBodies	= toCopy.m_rigidBodies;
	m_phantoms		= toCopy.m_phantoms;
	m_constraints	= toCopy.m_constraints;
	m_actions		= toCopy.m_actions;
	m_name			= toCopy.m_name;
	m_userData		= toCopy.m_userData;
	m_active		= toCopy.m_active;
}


hkpPhysicsSystem* hkpPhysicsSystem::clone(CloneConstraintMode cloneMode) const
{
	hkpPhysicsSystem* newSystem = new hkpPhysicsSystem();
	newSystem->m_name = m_name;
	newSystem->m_userData = m_userData;
	newSystem->m_active = m_active;

	newSystem->m_rigidBodies.setSize( m_rigidBodies.getSize() );
	newSystem->m_phantoms.setSize( m_phantoms.getSize() );
	newSystem->m_constraints.setSize( m_constraints.getSize() );
	newSystem->m_actions.setSize( m_actions.getSize() );

	// RigidBodies and Phantoms:
	for (int i = 0; i < m_rigidBodies.getSize(); ++i )
	{
		HK_ASSERT2(0x374076ab, m_rigidBodies[i] != HK_NULL, "You cannot have a NULL rigidbody pointer in a physics system - remove the element from the array instead." );
		newSystem->m_rigidBodies[i] = m_rigidBodies[i]->clone();
	}	
	for (int i = 0; i < m_phantoms.getSize(); ++i )
	{
		HK_ASSERT2(0x374076ac, m_phantoms[i] != HK_NULL, "You cannot have a NULL phantom pointer in a physics system - remove the element from the array instead." );
		newSystem->m_phantoms[i] = m_phantoms[i]->clone();
	}

	// Constraints and Actions.
	for (int i = 0; i < m_constraints.getSize(); ++i )
	{
		hkpConstraintInstance* c = m_constraints[i];

		HK_ASSERT2(0x374076ad, c != HK_NULL, "You cannot have a NULL constraint pointer in a physics system - remove the element from the array instead." );
		
		int newBodyAIndex = m_rigidBodies.indexOf( (hkpRigidBody*)( c->getEntityA() ) );
		HK_ASSERT2(0x374076ae, (newBodyAIndex > -1), "Rigidbodies are referenced by a constraint in the physics system that are not in the physics system.");
		hkpRigidBody* body_a = newSystem->m_rigidBodies[newBodyAIndex];

		hkpRigidBody* body_b;

		// Check if the constraint is connected to the world's fixed rigid body.
		if ( c->isConstrainedToWorld() )
		{
			body_b = HK_NULL;
		}
		else
		{
			int newBodyBIndex = m_rigidBodies.indexOf( (hkpRigidBody*)( c->getEntityB() ) );
			HK_ASSERT2(0x374076ae, (newBodyBIndex > -1), "Rigidbodies are referenced by a constraint in the physics system that are not in the physics system.");
			body_b = newSystem->m_rigidBodies[newBodyBIndex];
		}

		
		if (cloneMode == hkpPhysicsSystem::CLONE_SHALLOW_IF_NOT_CONSTRAINED_TO_WORLD) 
		{
			newSystem->m_constraints[i] = c->clone( body_a, body_b, hkpConstraintInstance::CLONE_SHALLOW_IF_NOT_CONSTRAINED_TO_WORLD );
		}
		else if (cloneMode ==hkpPhysicsSystem::CLONE_DEEP_WITH_MOTORS) 
		{
			newSystem->m_constraints[i] = c->clone( body_a, body_b, hkpConstraintInstance::CLONE_DATAS_WITH_MOTORS );
		}
		else if (cloneMode == hkpPhysicsSystem::CLONE_FORCE_SHALLOW) 
		{
			newSystem->m_constraints[i] = c->clone( body_a, body_b, hkpConstraintInstance::CLONE_FORCE_SHALLOW );
		}
	}
	
	for (int i = 0; i < m_actions.getSize(); ++i )
	{
		hkpAction* a = m_actions[i];
		if (!a) 
		{
			newSystem->m_actions[i] = HK_NULL; 
			continue;
		}

		hkArray<hkpEntity*> actionEntities;
		a->getEntities(actionEntities);

		hkArray<hkpEntity*> newRBs;
		int numE = actionEntities.getSize();
		newRBs.setSize(numE);
		int j = 0;
		for (j = 0; j < numE; ++j)
		{
			hkpRigidBody* rb = (hkpRigidBody*)actionEntities[j];
			int newBodyIndex = m_rigidBodies.indexOf( rb );
			HK_ASSERT2(0x374076ae, newBodyIndex >= 0, "Rigid bodies not in the physics system are referenced by an action in the physics system.");
			newRBs[j] = newSystem->m_rigidBodies[ newBodyIndex ];
		}

		hkArray<hkpPhantom*> actionPhantoms;
		a->getPhantoms(actionPhantoms);

		hkArray<hkpPhantom*> newPhantoms;
		int numP = actionPhantoms.getSize();
		newPhantoms.setSize(numP);
		for (j = 0; j < numP; ++j)
		{
			hkpPhantom* ph = (hkpPhantom*)actionPhantoms[j];
			int newPhantomIndex = m_phantoms.indexOf( ph );
			HK_ASSERT2(0x3cebe3c4, newPhantomIndex >= 0, "Phantoms not in the physics system are referenced by an action in the physics system.");
			newPhantoms[j] = newSystem->m_phantoms[ newPhantomIndex ];
		}

		newSystem->m_actions[i] = a->clone( newRBs, newPhantoms );
	}

	return newSystem;
}

void hkpPhysicsSystem::addRigidBody( hkpRigidBody* rb )
{
	if (rb)
	{
		rb->addReference();
		m_rigidBodies.pushBack(rb);
	}
}

void hkpPhysicsSystem::addPhantom(  hkpPhantom* p )
{
	if (p)
	{
		p->addReference();
		m_phantoms.pushBack(p);
	}
}

void hkpPhysicsSystem::addConstraint( hkpConstraintInstance* c )
{
	if (c)
	{
		c->addReference();
		m_constraints.pushBack(c);
	}
}

void hkpPhysicsSystem::addAction( hkpAction* a )
{
	if (a)
	{
		a->addReference();
		m_actions.pushBack(a);
	}
}

void hkpPhysicsSystem::removeRigidBody( int i )
{
	m_rigidBodies[i]->removeReference();
	m_rigidBodies.removeAt(i);
}

void hkpPhysicsSystem::removePhantom( int i )
{
	m_phantoms[i]->removeReference();
	m_phantoms.removeAt(i);
}

void hkpPhysicsSystem::removeConstraint( int i )
{
	m_constraints[i]->removeReference();
	m_constraints.removeAt(i);
}

void hkpPhysicsSystem::removeAction( int i )
{
	m_actions[i]->removeReference();
	m_actions.removeAt(i);
}

void hkpPhysicsSystem::removeNullPhantoms()
{
	// This is suboptimal, as this does multiple array-remove-and-copy.
	for (int i = m_phantoms.getSize()-1; i >= 0; i--)
	{
		if (m_phantoms[i] == HK_NULL)
		{
			m_phantoms.removeAtAndCopy(i);
		}
	}
}

static void tranformPhantom(hkpPhantom* phantom, const hkTransform& transformation)
{
 	const hkpPhantomType type = phantom->getType();

	switch(type)
	{
	case HK_PHANTOM_AABB:
		{
			hkpAabbPhantom* aabbPhantom = static_cast<hkpAabbPhantom*>(phantom);

			hkAabb aabb = aabbPhantom->getAabb();
			aabb.m_min.add(transformation.getTranslation());
			aabb.m_max.add(transformation.getTranslation());
			aabbPhantom->setAabb(aabb);

#if defined HK_DEBUG

			hkRotation rot = transformation.getRotation();
			if (rot.isOk() && !rot.isApproximatelyEqual(hkRotation::getIdentity()))
			{
				HK_WARN(0x0606a478, "Rotations of AABB phantoms is not supported. You would have to manaully handle this operation.");
			}

#endif // HK_DEBUG

		}
		break;

	case HK_PHANTOM_CACHING_SHAPE:	
	case HK_PHANTOM_SIMPLE_SHAPE:
		{
			hkpShapePhantom* shapePhantom = static_cast<hkpSimpleShapePhantom*>(phantom);
			hkTransform  tran = shapePhantom->getTransform();
			tran.setMulEq(transformation);
			shapePhantom->setTransform(tran);
		}
		break;

	default:
		{
			HK_WARN(0x0606a578, "Don't know how to transform this phantom");
		}
	}
}

void hkpPhysicsSystem::transform(const hkTransform& transformation)
{
	// Transform all the rigid bodies in this system
	for (int i = 0; i < m_rigidBodies.getSize(); i++)
	{
		hkTransform temptrans = transformation;
		temptrans.setMulEq(m_rigidBodies[i]->getTransform());
		m_rigidBodies[i]->setTransform(temptrans);
	}

	// Transform all the constraints
	for (int i = 0; i < m_constraints.getSize(); i++)
	{
		HK_ASSERT2(0x0606a678, m_constraints[i], "Constraint Instance must be initialized.");

		
		m_constraints[i]->transform( transformation );
	}
  
	// Transform all the phantoms
	for (int i = 0; i < m_phantoms.getSize(); i++)
	{
		tranformPhantom(m_phantoms[i], transformation); 
	}
	
	// Actions can not be transformed.
	// i.e. makes no sense logically to transform actions
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
