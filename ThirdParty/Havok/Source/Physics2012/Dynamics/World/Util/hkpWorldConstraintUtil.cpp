/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>

#if defined(HK_PLATFORM_SPU)
#	include <Common/Base/Spu/Dma/Manager/hkSpuDmaManager.h>
#	include <Physics2012/Dynamics/World/Simulation/Multithreaded/Spu/hkpSpuConfig.h>
#endif

#include <Physics2012/Dynamics/World/Util/hkpWorldConstraintUtil.h>

#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Dynamics/Constraint/Atom/hkpSimpleContactConstraintAtom.h>
#include <Physics2012/Dynamics/Constraint/hkpConstraintInstance.h>

#include <Physics2012/Dynamics/World/hkpWorld.h>
#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldOperationUtil.h>
#include <Physics2012/Dynamics/Collide/hkpResponseModifier.h>


#if !defined(HK_PLATFORM_SPU)

void hkpWorldConstraintUtil::addConstraint( hkpWorld* world, hkpConstraintInstance* constraint )
{
	HK_ACCESS_CHECK_WITH_PARENT( world, HK_ACCESS_RO, constraint->m_entities[0]->getSimulationIsland(), HK_ACCESS_RW);
	HK_ACCESS_CHECK_WITH_PARENT( world, HK_ACCESS_RO, constraint->m_entities[1]->getSimulationIsland(), HK_ACCESS_RW);

	HK_ASSERT2(0x312cc775 , HK_NULL == constraint->getOwner(), "A constraint cannot be added to the world twice.");
#ifdef HK_DEBUG
	for (int i = 0; i < 2; ++i)
	{
		HK_ASSERT2(0x1158af49, constraint->m_entities[i], "Both entities must be set before adding a constraint to the world.");
		HK_ASSERT2(0x476bb28b, constraint->m_entities[i]->getSimulationIsland(), "Both entities must be added to a world before adding the constraint." );
	}
#endif

	// add reference (kept by the owner)
	constraint->addReference();

	hkpConstraintData::ConstraintInfo info;	constraint->getData()->getConstraintInfo(info);

	hkConstraintInternal internalBuffer;
	{
		internalBuffer.m_constraint  = constraint;
		internalBuffer.m_entities[0] = constraint->m_entities[0];
		internalBuffer.m_entities[1] = constraint->m_entities[1];
		hkpModifierConstraintAtom* mod = constraint->m_constraintModifiers;
		internalBuffer.m_atoms       = (mod) ? mod             : info.m_atoms;
		internalBuffer.m_atomsSize   = (mod) ? mod->m_modifierAtomSize : hkUint16(info.m_sizeOfAllAtoms);
		internalBuffer.m_priority    = constraint->m_priority;
	}

	hkConstraintInternal* intern = &internalBuffer;
	HK_ASSERT2(0x77f2af97 , intern->m_entities[0]->getWorld() == intern->m_entities[1]->getWorld(),  "Both constraints must be added to the same world.");
	hkpWorldOperationUtil::mergeIslandsIfNeeded(intern->m_entities[0], intern->m_entities[1]);

	// choose the master entity
	{
		// in case of a non multithreaded environment we choose our master 
		// if several threads are running our choice must independent of the current master lists
#	if HK_CONFIG_THREAD != HK_CONFIG_MULTI_THREADED 
		if (intern->m_entities[0]->isFixed() == intern->m_entities[1]->isFixed())
		{
			// choose the one with more number of master constraints unless we have more than 8 masters
			int numMasters0 = intern->m_entities[0]->m_constraintsMaster.getSize();
			int numMasters1 = intern->m_entities[1]->m_constraintsMaster.getSize();
			if ( numMasters0 + numMasters1 < 8 )
			{
				intern->m_whoIsMaster =  numMasters0 >= numMasters1;
			}
			else
			{
				intern->m_whoIsMaster =  numMasters0 <= numMasters1;
			}
		}
		else
#	endif
		{
			// choose the unfixed one (if both are fixed/unfixed)
			intern->m_whoIsMaster = intern->m_entities[0]->isFixed();
		}
	}
	hkpEntity* masterEntity = intern->getMasterEntity();

	// mark the constraint as inserted into the world
	hkpSimulationIsland* island = masterEntity->getSimulationIsland();

	constraint->setOwner( island );
	island->m_numConstraints++;

	// update hkSimulationInformation (to adjust its constraint cache)
	{
		hkpConstraintAtom::AtomType atomType = intern->getAtoms()->getType();

		intern->m_constraintType  = hkpConstraintInstance::TYPE_NORMAL;
		intern->m_callbackRequest = hkpConstraintAtom::CALLBACK_REQUEST_NONE;

		// Skip transform atoms so we can see the pulley / wheel friction atom that follows and disable them on SPU.
		switch (atomType)
		{
		case hkpConstraintAtom::TYPE_SET_LOCAL_TRANSFORMS:
			atomType = static_cast<hkpSetLocalTransformsConstraintAtom*>(intern->getAtoms())->next()->getType();
			break;
		case hkpConstraintAtom::TYPE_SET_LOCAL_TRANSLATIONS:
			atomType = static_cast<hkpSetLocalTranslationsConstraintAtom*>(intern->getAtoms())->next()->getType();
			break;
		case hkpConstraintAtom::TYPE_SET_LOCAL_ROTATIONS:
			atomType = static_cast<hkpSetLocalRotationsConstraintAtom*>(intern->getAtoms())->next()->getType();
			break;
		default:
			break;
		}

		// Ensure we're checking the constraint, rather than a modifier. Modifiers will be checked later.
		if( atomType >= hkpConstraintAtom::FIRST_MODIFIER_TYPE && atomType <= hkpConstraintAtom::LAST_MODIFIER_TYPE )
		{
			hkpModifierConstraintAtom* modifier = reinterpret_cast<hkpModifierConstraintAtom*>( intern->getAtoms() );
			atomType = modifier->m_child->getType();
		}

		if ( atomType == hkpConstraintAtom::TYPE_BRIDGE )
		{
			intern->m_callbackRequest |= hkpConstraintAtom::CALLBACK_REQUEST_SETUP_PPU_ONLY;

			if (constraint->getData()->isBuildJacobianCallbackRequired())
			{
				intern->m_callbackRequest |= hkpConstraintAtom::CALLBACK_REQUEST_SETUP_CALLBACK;
			}

			intern->m_constraintType = constraint->getType();
		}

		if ( atomType == hkpConstraintAtom::TYPE_PULLEY )
		{
			intern->m_callbackRequest |= hkpConstraintAtom::CALLBACK_REQUEST_SETUP_PPU_ONLY;
		}
		
		if ( atomType == hkpConstraintAtom::TYPE_WHEEL_FRICTION )
		{
#ifdef HK_PLATFORM_PS3
			HK_WARN_ONCE(0x589d57fb, "The per-wheel friction solver is not supported on SPU. Using it may cause a huge performance loss.");
#endif
			intern->m_callbackRequest |= hkpConstraintAtom::CALLBACK_REQUEST_SETUP_PPU_ONLY;
			intern->m_constraintType = hkpConstraintInstance::TYPE_DISABLE_SPU;
		}

		hkUint8 usedModifierFlags = 0;
		if ( constraint->m_constraintModifiers)
		{
			intern->m_callbackRequest |= hkpModifierConstraintAtom::addAllModifierDataToConstraintInfo( constraint->m_constraintModifiers, info, usedModifierFlags );
		}

		// Allocate additional space to allow response modifiers to be added in contact callbacks.

		hkpConstraintAtom::AtomType terminalAtomType = hkpWorldConstraintUtil::getTerminalAtom(intern)->getType();
		if ( terminalAtomType == hkpConstraintAtom::TYPE_CONTACT )
		{
			// We need to make extra space for modifiers, as noted in either of the bodies.
			hkUint8 flags = constraint->m_entities[0]->m_responseModifierFlags | constraint->m_entities[1]->m_responseModifierFlags;
			// But don't make extra space for those modifiers which have already been added.
			flags -= flags & usedModifierFlags;
			info.m_sizeOfSchemas += hkpResponseModifier::getAdditionalSchemaSize( flags );
		}

		intern->clearConstraintInfo();
		constraint->m_internal = intern;	// direct internal to our temporary buffer, will be corrected later
		constraint->getOwner()->addConstraintInfo(constraint, info);
	}

	
	//
	// insert m_internal to the entity's list and preserve the constraint order
	//
	int insertPos;
	{
		hkSmallArray<hkConstraintInternal>& masters = masterEntity->m_constraintsMaster;
		int lim = masters.getSize();
		for (insertPos = 0; insertPos < lim; insertPos++)
		{
			hkConstraintInternal& currentMaster = masters[insertPos];
			// add constraint after the last other constraint of the same or lesser priority
			if (intern->m_priority < currentMaster.m_priority ) { break; }

#		if HK_CONFIG_THREAD == HK_CONFIG_MULTI_THREADED 
			if (intern->m_priority > currentMaster.m_priority ) { continue; }

			// in case of a multithreaded environment we have to make sure that all constraints
			// are processed in a deterministic way. We achieve this by making sure that the
			// order of constraints in the currentMaster list is independent from the order constraints are added.
			//
			// to guarantee deterministic order, we simply order the constraint by their uids.
			// And then by their connected entities' uids.

			if (intern->m_constraint->m_uid < currentMaster.m_constraint->m_uid) { break; } 	 
			if (intern->m_constraint->m_uid > currentMaster.m_constraint->m_uid) { continue; } 	 

			const hkpEntity* internOtherEntity = intern->getOtherEntity(masterEntity);
			const hkpEntity* currentMasterOtherEntity = currentMaster.getOtherEntity(masterEntity);

			if (internOtherEntity->getUid() < currentMasterOtherEntity->getUid()) { break; } 	 
#		endif
		}

		int firstIndexToRelink = ( masters.getSize() < masters.getCapacity() ) ? insertPos : 0;

		// move the instance of hkConstraintInternal
		// and relink our masters
		masters.insertAt( insertPos, *intern );

		for ( int m = firstIndexToRelink; m < masters.getSize(); m++ )
		{
			hkConstraintInternal* ci = &masters[m];
			ci->m_constraint->m_internal = ci;
		}

		// refresh the pointer 
		intern = constraint->m_internal;
	}

	//    
	// allocate and reserve space in our constraintRuntime
	//
	{
		hkSmallArray<hkConstraintInternal>& masters = masterEntity->m_constraintsMaster;
		hkpConstraintData::RuntimeInfo rInfo;
		intern->m_constraint->getData()->getRuntimeInfo( constraint->m_wantRuntime, rInfo );
		rInfo.m_sizeOfExternalRuntime = HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT, rInfo.m_sizeOfExternalRuntime);
		intern->m_runtimeSize = (hkUint16)( rInfo.m_sizeOfExternalRuntime );
		// todo: think about that assert and where it belongs
		//HK_ASSERT2(0x589d57fa, hkUlong(intern->m_runtimeSize) <= HK_SPU_CONSTRAINT_RUNTIME_BUFFER_SIZE, "Runtime doesn't fit into static buffer on spu.");

		if ( rInfo.m_sizeOfExternalRuntime )
		{
			hkArray<hkUint8>& rt = masterEntity->m_constraintRuntime;
			int oldSize = rt.getSize();
			hkUint8* oldStart = rt.begin();
			rt.reserveExactly( oldSize + rInfo.m_sizeOfExternalRuntime );
			rt.setSizeUnchecked( oldSize + rInfo.m_sizeOfExternalRuntime );
			hkUint8* newStart = rt.begin();
			hkUint8* insertPoint = newStart;
			hkUlong offset = hkGetByteOffset( oldStart, newStart );

			// fix pointers of all constraints till insertPos
			int p;
			for (p = 0; p < insertPos; p++)
			{
				hkConstraintInternal* ci = &masters[p];
				if ( ci->m_runtime )
				{
					ci->m_runtime = hkAddByteOffset( ci->m_runtime, offset );
					insertPoint = (hkUint8*)hkAddByteOffset( ci->m_runtime, ci->m_runtimeSize );
				}
			}
			// insert our runtime data
			{
				hkUint8* restEnd = rt.end() - intern->m_runtimeSize;
				hkUlong restSize = hkGetByteOffset( insertPoint, restEnd );
				hkString::memMove( insertPoint + intern->m_runtimeSize, insertPoint, int(restSize) );
				intern->m_runtime = reinterpret_cast<hkpConstraintRuntime*>(insertPoint);
				p++;
			}

			// fix pointers of the rest of the constraints
			offset += intern->m_runtimeSize;
			for (; p < masters.getSize(); p++)
			{
				hkConstraintInternal* ci = &masters[p];
				ci->m_runtime = (ci->m_runtime) ? hkAddByteOffset( ci->m_runtime, offset ): HK_NULL;
			}
		}
		else
		{
			intern->m_runtime = HK_NULL;
		}
		intern->m_constraint->m_data->addInstance( intern->m_runtime, intern->m_runtimeSize );
	}

	// just update the slave's pointer to the constraint
	hkpEntity* slaveEntity = intern->getSlaveEntity();
	{
		intern->m_slaveIndex = hkObjectIndex( slaveEntity->m_constraintsSlave.getSize() );
		slaveEntity->m_constraintsSlave.pushBack( constraint );
	}
}


void hkpWorldConstraintUtil::removeConstraint( hkpConstraintInstance* constraint )
{
	HK_ASSERT2(0x654d83cc, HK_NULL != constraint->getOwner(), "Trying to remove a constraint that has not been added to a world.");

	HK_ON_DEBUG_MULTI_THREADING( hkpWorld* world = constraint->m_entities[0]->getWorld() );
	HK_ACCESS_CHECK_WITH_PARENT( world, HK_ACCESS_RO, constraint->m_entities[0]->getSimulationIsland(), HK_ACCESS_RW);
	HK_ACCESS_CHECK_WITH_PARENT( world, HK_ACCESS_RO, constraint->m_entities[1]->getSimulationIsland(), HK_ACCESS_RW);

	hkConstraintInternal* intern = constraint->m_internal;

	//
	// remove the constraint from the island; reset island's cache, reset constraints' owner
	//
	
	// request island split check
	hkpSimulationIsland* island = static_cast<hkpSimulationIsland*>(constraint->getOwner());
	{
		island->m_splitCheckRequested = true;
		HK_ACCESS_CHECK_WITH_PARENT( island->m_world, HK_ACCESS_RO, island, HK_ACCESS_RW );
	}

	// update hkSimulationInformation (to adjust its constraint cache)
	{
		hkpConstraintInfo info;
		intern->getConstraintInfo(info);
		constraint->getOwner()->subConstraintInfo(constraint, info);
	}
	island->m_numConstraints--;
	constraint->setOwner( HK_NULL );

	// remove the constraint from the slave body 
	{
		hkpEntity* slaveEntity = intern->getSlaveEntity();
		hkObjectIndex slaveIndex = intern->m_slaveIndex;

		hkArray<hkpConstraintInstance*>& slaves = slaveEntity->m_constraintsSlave;
		hkpConstraintInstance* lastConstraint = slaves.back();
			// check whether we are removing the right thing
		HK_ASSERT( 0Xf0346778, slaves[slaveIndex] == constraint );
		slaves[slaveIndex] = lastConstraint;
		slaves.popBack();
		lastConstraint->m_internal->m_slaveIndex = slaveIndex;
	}

	hkpEntity* masterEntity = intern->getMasterEntity();

	// remove the runtime data from the master body
	hkUlong runtimeOffset = 0;
	if ( intern->m_runtime )
	{
		hkConstraintInternal* ci = intern;
		runtimeOffset = -ci->m_runtimeSize;
		hkpConstraintRuntime* toDeleteBegin = ci->m_runtime;
		hkpConstraintRuntime* toDeleteEnd   = hkAddByteOffset( ci->m_runtime, ci->m_runtimeSize );
		hkUint8* totalEnd      = masterEntity->m_constraintRuntime.end();
		hkUlong restLen = hkGetByteOffset( toDeleteEnd, totalEnd );
		hkString::memMove( toDeleteBegin, toDeleteEnd,  int(restLen) );
		masterEntity->m_constraintRuntime.popBack( ci->m_runtimeSize );
		ci->m_runtime = HK_NULL; 
	}

	// remove the constraint from the master body (it's gotta be there !!)
	// and relink the rest of the entries
	{
		hkConstraintInternal* ci = intern;

		constraint->m_internal = HK_NULL;

		hkConstraintInternal* back = &masterEntity->m_constraintsMaster.back();

		// preserve the order of constraints
		while( ci < back )
		{
			ci[0] = ci[1];
			ci->m_constraint->m_internal = ci;
			ci->m_runtime = (ci->m_runtime) ? hkAddByteOffset(ci->m_runtime, runtimeOffset): HK_NULL;
			ci++;
		}
		masterEntity->m_constraintsMaster.popBack();
	}

	// remove reference (been kept by the owner)
	constraint->removeReference();
}

hkpConstraintInstance* hkpWorldConstraintUtil::getConstraint( const hkpEntity* entityA, const hkpEntity* entityB)
{
	HK_ON_DEBUG_MULTI_THREADING( hkpWorld* world = entityA->getWorld() );
	HK_ACCESS_CHECK_WITH_PARENT( world, HK_ACCESS_IGNORE, entityA->getSimulationIsland(), HK_ACCESS_RO);
	HK_ACCESS_CHECK_WITH_PARENT( world, HK_ACCESS_IGNORE, entityB->getSimulationIsland(), HK_ACCESS_RO);

	const hkpEntity* entities[] = { entityA, entityB };

	for (int i = 0; i < 2; i++)
	{
		const hkpEntity* masterEntity = entities[i];
		const hkpEntity* otherEntity  = entities[1-i];

		for (int c = 0; c < masterEntity->m_constraintsMaster.getSize(); c++)
		{
			if (masterEntity->m_constraintsMaster[c].getOtherEntity(masterEntity) == otherEntity)
			{
				// one or more constraints exist
				return masterEntity->m_constraintsMaster[c].m_constraint;
			}
		}
	}
	return HK_NULL;
}



void hkpWorldConstraintUtil::addModifier( hkpConstraintInstance* instance, hkpConstraintOwner& constraintOwner, hkpModifierConstraintAtom* newModifier )
{
  	constraintOwner.checkAccessRw();

	HK_ASSERT2( 0x134ef55e, findModifier( instance, newModifier->getType() ) == HK_NULL, \
		"You are trying to add a modifier to a constraint that already has a modifier of that type. Use findModifier() to check this first." );


	hkpModifierConstraintAtom*& firstModifier = instance->m_constraintModifiers;
	hkpModifierConstraintAtom* lastModifier = hkpWorldConstraintUtil::findLastModifier( instance );
	hkConstraintInternal* cInternal = instance->m_internal;

	if ( newModifier->getType() == hkpConstraintAtom::TYPE_MODIFIER_MOVING_SURFACE && lastModifier != HK_NULL )
	{
		//
		// only insert new container at the end of the modifiers-list if new modifier is of type "moving surface"
		// and if there are already modifiers attached to this constraint
		//
		newModifier->m_child     = lastModifier->m_child;
		newModifier->m_childSize = lastModifier->m_childSize;

		// insert new modifier between terminal atom and the current "last modifier"
		lastModifier->m_child     =  newModifier;
		lastModifier->m_childSize =  newModifier->m_modifierAtomSize;
	}
	else if (firstModifier && firstModifier->getType() == hkpConstraintAtom::TYPE_MODIFIER_IGNORE_CONSTRAINT)
	{
		// Make sure the ignore modifier is always at the top of the list 
		//
		// Insert the new elemenent on the second place, right after the ignore modifier
		//
		newModifier->m_child     = firstModifier->m_child;
		newModifier->m_childSize = firstModifier->m_childSize;

		firstModifier->m_child     =  newModifier;
		firstModifier->m_childSize =  newModifier->m_modifierAtomSize;

	}
	else
	{
		// add new modifier at the begin of our modifier list
		instance->m_constraintModifiers = newModifier;


		if ( cInternal )
		{
			newModifier->m_child     = cInternal->m_atoms;
			newModifier->m_childSize = cInternal->m_atomsSize;
			cInternal->m_atoms       =  newModifier;
			cInternal->m_atomsSize   =  newModifier->m_modifierAtomSize;
		}
		else
		{
			hkpConstraintData::ConstraintInfo cinfo; instance->getData()->getConstraintInfo( cinfo );
			newModifier->m_child     = cinfo.m_atoms;
			newModifier->m_childSize = hkUint16(cinfo.m_sizeOfAllAtoms);
		}
	}

	//
	//	Update constraint info in constraint owner and master list only if the constraint is added to the world
	//
	if ( cInternal )
	{
		hkpConstraintInfo cinfo;		cinfo.clear();
		hkUint8 flag = 0;
		int callbackRequest = newModifier->addModifierDataToConstraintInfo( cinfo, flag );
		constraintOwner.addConstraintInfo( instance, cinfo );

		// We may have made space for this modifier already.
		flag &= ( instance->m_entities[0]->m_responseModifierFlags | instance->m_entities[1]->m_responseModifierFlags );
		hkUint16 extraSpace = hkpResponseModifier::getAdditionalSchemaSize( flag );
		constraintOwner.m_constraintInfo.m_sizeOfSchemas -= extraSpace;
		instance->m_internal->m_sizeOfSchemas = instance->m_internal->m_sizeOfSchemas - extraSpace;		

		if ( instance->m_internal )
		{
			instance->m_internal->m_callbackRequest |= callbackRequest;
		}
	}
}



void hkpWorldConstraintUtil::removeAndDeleteModifier( hkpConstraintInstance* instance, hkpConstraintOwner& constraintOwner, hkpConstraintAtom::AtomType type )
{
	constraintOwner.checkAccessRw();

	hkpModifierConstraintAtom* modifier = instance->m_constraintModifiers;

	if ( !modifier )
	{
		return;
	}
	hkConstraintInternal* cInternal = instance->m_internal;


	//
	//	Check whether we are the first modifier
	//
	{
		HK_ASSERT( 0xf0323454, modifier->isModifierType());

		hkpConstraintAtom::AtomType modType = modifier->getType();
		if ( modType == type)
		{
			hkpConstraintAtom* child = modifier->m_child;
			if ( cInternal )
			{
				cInternal->m_atoms     = modifier->m_child;
				cInternal->m_atomsSize = modifier->m_childSize;
			}

			if ( child->isModifierType() )
			{
				instance->m_constraintModifiers = static_cast<hkpModifierConstraintAtom*>(child);
			}
			else
			{
				instance->m_constraintModifiers = HK_NULL;
			}

			goto UPDATE_CONSTRAINT_INFO;
		}
	}

	{
	    hkpModifierConstraintAtom* father = modifier;
	    hkpConstraintAtom* atom = modifier->m_child;
    
	    while ( 1 )
	    {
		    // abort if we reached the constraint's original atom
		    if ( !atom->isModifierType() )
		    {
			    return;
		    }
		    modifier = static_cast<hkpModifierConstraintAtom*>(atom);
    
		    hkpConstraintAtom::AtomType modType = modifier->getType();
    
		    if ( modType == type )
		    {
			    father->m_child     = modifier->m_child;
			    father->m_childSize = modifier->m_childSize;
			    goto UPDATE_CONSTRAINT_INFO;
		    }
		    father  = modifier;
		    atom   = modifier->m_child;
	    }
	}

UPDATE_CONSTRAINT_INFO:
	//
	//	Update constraint info in constraintOwner and master list
	//
	if (cInternal)
	{
		hkpConstraintInfo cinfo;	cinfo.clear();
		hkUint8 flag = 0;
		modifier->addModifierDataToConstraintInfo( cinfo, flag );
		// We may need to keep space for this modifier.
		flag &= ( instance->m_entities[0]->m_responseModifierFlags | instance->m_entities[1]->m_responseModifierFlags );
		hkUint16 extraSpace = hkpResponseModifier::getAdditionalSchemaSize( flag );
		constraintOwner.m_constraintInfo.m_sizeOfSchemas += extraSpace;
		instance->m_internal->m_sizeOfSchemas = instance->m_internal->m_sizeOfSchemas + extraSpace;

		constraintOwner.subConstraintInfo( cInternal->m_constraint, cinfo );
	}

	HK_MEMORY_TRACKER_DELETE_OBJECT(modifier)
	hkDeallocateChunk<void>(modifier, modifier->m_modifierAtomSize, HK_MEMORY_CLASS_CONSTRAINT );
}



hkpModifierConstraintAtom* hkpWorldConstraintUtil::findModifier( hkpConstraintInstance* instance, hkpConstraintAtom::AtomType type )
{
	HK_ON_DEBUG_MULTI_THREADING( hkpWorld* world = instance->m_entities[0]->getWorld() );
	HK_ACCESS_CHECK_WITH_PARENT( world, HK_ACCESS_RO, instance->getSimulationIsland(), HK_ACCESS_RO);

	hkpModifierConstraintAtom* atom = instance->m_constraintModifiers;
	if (!atom)
	{
		return HK_NULL;
	}

	while ( 1 )
	{
		// abort if we reached the constraint's original atom
		hkpConstraintAtom::AtomType modType = atom->getType();

		if ( modType == type )
		{
			return atom;
		}
		if ( !atom->m_child->isModifierType() )
		{
			return HK_NULL;
		}

		atom = static_cast<hkpModifierConstraintAtom*>(atom->m_child);
	}

}



hkpModifierConstraintAtom* hkpWorldConstraintUtil::findLastModifier( hkpConstraintInstance* instance )
{
	HK_ON_DEBUG_MULTI_THREADING( hkpWorld* world = instance->getEntityA()->getWorld());
	HK_ACCESS_CHECK_WITH_PARENT( world, HK_ACCESS_RO, instance->getSimulationIsland(), HK_ACCESS_RO);

	hkpModifierConstraintAtom* atom = instance->m_constraintModifiers;

		//
		// return HK_NULL if no modifier present
		//
	if ( !atom )
	{
		return HK_NULL;
	}

		//
		// search modifier list for last modifier
		//
	while ( 1 )
	{
		hkpConstraintAtom* childAtom     = atom->m_child;

		// abort if we reached the constraint's terminal atom
		if ( !childAtom->isModifierType() )
		{
			return atom;
		}

		atom = static_cast<hkpModifierConstraintAtom*>(childAtom);
	}
}

#endif

void hkpWorldConstraintUtil::updateFatherOfMovedAtom( hkpConstraintInstance* instance, const hkpConstraintAtom* oldAtom, const hkpConstraintAtom* updatedAtom, int updatedSizeOfAtom )
{
	HK_ASSERT2(0xaf83fe64, updatedAtom, "Updated atom is invalid.");

	//
	// update constraint internal if no modifiers are attached to constraint
	//
	hkConstraintInternal* cInternal = instance->m_internal;
	if ( !instance->m_constraintModifiers )
	{
		if (cInternal)
		{
			HK_ASSERT( 0xf0e3ed45, oldAtom == cInternal->m_atoms);
			cInternal->m_atoms     = const_cast<hkpConstraintAtom*>( updatedAtom );
			cInternal->m_atomsSize = hkUint16(updatedSizeOfAtom);
		}
		return;
	}

	if ( instance->m_constraintModifiers == oldAtom )
	{
		instance->m_constraintModifiers = const_cast<hkpModifierConstraintAtom*>(static_cast<const hkpModifierConstraintAtom*>(updatedAtom));
		if (cInternal)
		{
		    cInternal->m_atoms     = const_cast<hkpConstraintAtom*>( updatedAtom );
		    cInternal->m_atomsSize = hkUint16(updatedSizeOfAtom);
		}
		return;
	}

	//
	// search modifier list for last modifier
	//
#if defined(HK_PLATFORM_SPU)
	HK_ALIGN16( hkUint8 modifierBuffer[HK_NEXT_MULTIPLE_OF(16, sizeof(hkpModifierConstraintAtom))] );
	hkpModifierConstraintAtom* localModifier = reinterpret_cast<hkpModifierConstraintAtom*>( &modifierBuffer[0] );
#else
	hkpModifierConstraintAtom* localModifier = instance->m_constraintModifiers;
#endif

	hkpModifierConstraintAtom* modifier = instance->m_constraintModifiers;
	while ( 1 )
	{
#if defined(HK_PLATFORM_SPU)
		hkSpuDmaManager::getFromMainMemoryAndWaitForCompletion(localModifier, modifier, sizeof(hkpModifierConstraintAtom), hkSpuDmaManager::READ_COPY);
		HK_SPU_DMA_PERFORM_FINAL_CHECKS                       (modifier, localModifier, sizeof(hkpModifierConstraintAtom));
#else
		localModifier = modifier;
#endif

		hkpConstraintAtom* childAtom = localModifier->m_child;

		// abort if we reached the constraint's original atom
		if ( childAtom == oldAtom )
		{
			//
			// update last modifier
			//
			localModifier->m_child     = const_cast<hkpConstraintAtom*>( updatedAtom );
			localModifier->m_childSize = hkUint16(updatedSizeOfAtom);

#if defined(HK_PLATFORM_SPU)
			hkSpuDmaManager::putToMainMemoryAndWaitForCompletion(modifier, localModifier, sizeof(hkpModifierConstraintAtom), hkSpuDmaManager::WRITE_NEW);
			HK_SPU_DMA_PERFORM_FINAL_CHECKS                     (modifier, localModifier, sizeof(hkpModifierConstraintAtom));
#endif
			break;
		}
		HK_ASSERT2( 0xf0323454, childAtom->isModifierType(), "Internal inconsistencies with constraint modifiers" );
		modifier = static_cast<hkpModifierConstraintAtom*>(childAtom);
	}

}

#if !defined(HK_COMPILER_MWERKS)
// Metroworks compilers use a post-attribute for alignment.  To make the code cleaner,
// class alignment (i.e. HK_ALIGNCLASS) has been disabled metroworks compilers.
HK_COMPILE_TIME_ASSERT( (sizeof(hkpSoftContactModifierConstraintAtom)    & 0xf) == 0);
HK_COMPILE_TIME_ASSERT( (sizeof(hkpMassChangerModifierConstraintAtom)    & 0xf) == 0);
HK_COMPILE_TIME_ASSERT( (sizeof(hkpViscousSurfaceModifierConstraintAtom) & 0xf) == 0);
HK_COMPILE_TIME_ASSERT( (sizeof(hkpMovingSurfaceModifierConstraintAtom)  & 0xf) == 0);
#endif

//HK_COMPILE_TIME_ASSERT( (HK_OFFSET_OF(hkMovingSurfaceModifierAtomContainer, m_modifier.m_velocity[0]) & 0xf) == 0); // VS2005 does not like this...

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
