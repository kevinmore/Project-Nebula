/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Compat/hkCompat.h>
#include <Common/Serialize/hkSerialize.h>
#include <Common/Compat/Deprecated/Version/hkVersionCheckingUtilsOld.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>
#include <Common/Base/Config/hkConfigBranches.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/Base/System/Io/Writer/hkStreamWriter.h>
#include <Common/Serialize/Data/Dict/hkDataObjectDict.h>
#include <Common/Serialize/Data/hkDataObjectImpl.h>
#include <Common/Serialize/Data/Native/hkDataObjectNative.h>
#include <Common/Compat/Deprecated/Util/hkRenamedClassNameRegistry.h>
#include <Common/Serialize/Version/hkVersionPatchManager.h>

namespace
{
	struct ActionFromClassName : public hkStringMap<const hkVersionRegistry::ClassAction*>
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_BASE, ActionFromClassName);

		ActionFromClassName(
			const hkVersionRegistry::UpdateDescription& descHead,
			const hkClassNameRegistry& oldClassesIn )
		{
			const hkVersionRegistry::UpdateDescription* desc = &descHead;
			while( desc )
			{
				if( desc->m_actions )
				{
					m_actionChunks.insertAt(0, desc->m_actions);
					m_descFromActionChunk.insert(desc->m_actions, desc);
				}
				desc = desc->m_next;
			}
			hkPointerMap<const hkVersionRegistry::ClassAction*, hkBool32> flagFromActionPointer;
			hkArray<const hkClass*> classes;
			oldClassesIn.getClasses(classes);
			for( int i = 0; i < classes.getSize(); ++i )
			{
				const hkVersionRegistry::ClassAction* action = findFirstActionForClass( *classes[i] );
				if( action )
				{
					this->insert( classes[i]->getName(), action );
					flagFromActionPointer.insert(action, true);
				}
			}
			for( int i = 0; i < m_actionChunks.getSize(); ++i )
			{
				const hkVersionRegistry::ClassAction* action = m_actionChunks[i];
				for( ; action->oldClassName != HK_NULL; ++action )
				{
					if( flagFromActionPointer.hasKey(action) )
					{
						m_actions.expandOne() = action;
					}
					else
					{
						m_junkActions.expandOne() = action;
					}
				}
			}
		}

		const hkArray<const hkVersionRegistry::ClassAction*>& getActions()
		{
			return m_actions;
		}

		const hkArray<const hkVersionRegistry::ClassAction*>& getJunkActions()
		{
			return m_junkActions;
		}

		const hkClass* getClassForAction(const hkVersionRegistry::ClassAction* actionIn)
		{
			HK_ASSERT(0x41ee9db4, actionIn);
			// the first match for any base class wins
			for( int i = 0; i < m_actionChunks.getSize(); ++i )
			{
				const hkVersionRegistry::ClassAction* action = m_actionChunks[i];
				for( ; action->oldClassName != HK_NULL; ++action )
				{
					if( actionIn == action )
					{
						const hkVersionRegistry::UpdateDescription* desc = HK_NULL;
						HK_ON_DEBUG(hkResult res =) m_descFromActionChunk.get(m_actionChunks[i], &desc);
						HK_ASSERT(0x6c54e208, res == HK_SUCCESS);
						const char* className = actionIn->oldClassName;
						// check renames
						if( desc->m_renames )
						{
							for( int j = 0; desc->m_renames[j].oldName != HK_NULL; ++j )
							{
								if( hkString::strCmp(className, desc->m_renames[j].oldName) == 0 )
								{
									className = desc->m_renames[j].newName;
									break;
								}
							}
						}
						return desc->m_newClassRegistry ? desc->m_newClassRegistry->getClassByName(className) : HK_NULL;
					}
				}
			}
			return HK_NULL;
		}

	private:

		const hkVersionRegistry::ClassAction* findFirstActionForClass( const hkClass& classIn ) const
		{
			// we match all names in the hierarchy
			hkStringMap<const hkClass*> hierarchyNames;
			{
				const hkClass* c = &classIn;
				while(  c != HK_NULL )
				{
					hierarchyNames.insert(c->getName(), c);
					c = c->getParent();
				}
			}
			// the first match for any base class wins
			for( int i = 0; i < m_actionChunks.getSize(); ++i )
			{
				const hkVersionRegistry::ClassAction* action = m_actionChunks[i];
				for( ; action->oldClassName != HK_NULL; ++action )
				{
					if( const hkClass* c = hierarchyNames.getWithDefault( action->oldClassName, HK_NULL ) )
					{
						//			HK_ON_DEBUG(hkUint32 loadedSig = c->getSignature());
						//			HK_ASSERT( 0x786cb087, (action->versionFlags & VERSION_REMOVED) || loadedSig == action->oldSignature );
						return action;
					}
				}
			}
			return HK_NULL;
		}
	private:
		hkArray<const hkVersionRegistry::ClassAction*> m_actionChunks;
		hkPointerMap<const hkVersionRegistry::ClassAction*, const hkVersionRegistry::UpdateDescription*> m_descFromActionChunk;
		hkArray<const hkVersionRegistry::ClassAction*> m_actions;
		hkArray<const hkVersionRegistry::ClassAction*> m_junkActions;
	};

	HK_FORCE_INLINE hkBool32 _compareClass( const hkClass* a, const hkClass* b )
	{
		return hkString::strCmp(a->getName(), b->getName()) < 0;
	}

	int NOT(int i) { return !i; }

	static int containsVariants(const hkClass& klass)
	{
		for( int i = 0; i < klass.getNumMembers(); ++i )
		{
			const hkClassMember& mem = klass.getMember(i);
			switch( mem.getType() )
			{
				case hkClassMember::TYPE_VARIANT:
				{
					return 1;
				}
				case hkClassMember::TYPE_ARRAY:
				case hkClassMember::TYPE_SIMPLEARRAY:
				{
					if( mem.getSubType() == hkClassMember::TYPE_VARIANT )
					{
						return 1;
					}
					else if( mem.getSubType() != hkClassMember::TYPE_STRUCT )
					{
						break;
					}
					// struct falls through
				}
				case hkClassMember::TYPE_STRUCT:
				{
					if( containsVariants(mem.getStructClass()) )
					{
						return 1;
					}
					break;
				}
				default:
				{
					// skip
				}
			}
		}
		return 0;
	}

	static int containsHomogeneousArray(const hkClass& klass)
	{
		for( int i = 0; i < klass.getNumMembers(); ++i )
		{
			const hkClassMember& mem = klass.getMember(i);
			switch( mem.getType() )
			{
				case hkClassMember::TYPE_HOMOGENEOUSARRAY:
				{
					return 1;
				}
				case hkClassMember::TYPE_ARRAY:
				case hkClassMember::TYPE_SIMPLEARRAY:
				{
					if( mem.getSubType() != hkClassMember::TYPE_STRUCT )
					{
						break;
					}
					// struct falls through
				}
				case hkClassMember::TYPE_STRUCT:
				{
					if( mem.getClass() && containsHomogeneousArray(mem.getStructClass()) )
					{
						return 1;
					}
					break;
				}
				default:
				{
					// skip
				}
			}
		}
		return 0;
	}

	// COM-629, c-string -> hkStringPtr, hkVariant->hkRefVariant, hkSimpleArray -> hkArray
	static HK_FORCE_INLINE hkBool32 areMembersCompatible(const hkClassMember& src, const hkClassMember& dst)
	{
		return ( src.getType() == dst.getType()
				&& ( src.getSubType() == dst.getSubType()
					|| src.getSubType() == hkClassMember::TYPE_VOID
					|| dst.getSubType() == hkClassMember::TYPE_VOID
					|| src.getType() == hkClassMember::TYPE_ENUM ) )
			|| ( src.getType() == hkClassMember::TYPE_ZERO && dst.getType() != hkClassMember::TYPE_ZERO )
			|| ( src.getType() != hkClassMember::TYPE_ZERO && dst.getType() == hkClassMember::TYPE_ZERO )
			// COM-629, c-string -> hkStringPtr
			|| ( src.getType() == hkClassMember::TYPE_CSTRING && dst.getType() == hkClassMember::TYPE_STRINGPTR )
			// COM-629, hkVariant -> hkRefVariant (hkRefPtr<hkReferencedObject>)
			|| ( src.getType() == hkClassMember::TYPE_VARIANT
					&& dst.getType() == hkClassMember::TYPE_POINTER
					&& dst.getSubType() == hkClassMember::TYPE_STRUCT
					&& dst.getClass() && hkString::strCmp(hkReferencedObjectClass.getName(), dst.getStructClass().getName()) == 0 )
			// COM-629, hkSimpleArray -> hkArray, and array elements
			|| ( ( src.getType() == dst.getType() || ( src.getType() == hkClassMember::TYPE_SIMPLEARRAY && dst.getType() == hkClassMember::TYPE_ARRAY ) )
					&& ( src.getSubType() == dst.getSubType()
						// COM-629, array of c-string -> array of hkSrtingPtr
						|| ( src.getSubType() == hkClassMember::TYPE_CSTRING && dst.getSubType() == hkClassMember::TYPE_STRINGPTR )
						// COM-629, array of hkVariant -> array of hkRefVariant (hkRefPtr<hkReferencedObject>)
						|| ( src.getSubType() == hkClassMember::TYPE_VARIANT && dst.getSubType() == hkClassMember::TYPE_POINTER
							&& dst.getClass() && hkString::strCmp(hkReferencedObjectClass.getName(), dst.getStructClass().getName()) == 0 ) ) );
	}

	typedef hkStringMap<const char*> StringMap;
	inline hkBool32 equalNames(const char* oldClassName, const char* newClassName, const StringMap& newFromOldNameMap)
	{
		const char* renamedClass = newFromOldNameMap.getWithDefault(oldClassName, oldClassName);
		return (hkString::strCmp(renamedClass, newClassName) == 0);
	}
}

hkResult HK_CALL hkVersionCheckingUtils::verifyUpdateDescription(
	hkOstream& report,
	const hkClassNameRegistry& oldClassReg,
	const hkClassNameRegistry& newClassReg,
	const hkVersionRegistry::UpdateDescription& updateDescriptionHead,
	Flags flags, const char** ignoredPrefixes /* = HK_NULL */, int numIgnoredPrefixes /* = 0 */ )
{
	ActionFromClassName actionFromOldName(updateDescriptionHead, oldClassReg);
	hkRenamedClassNameRegistry newClassFromOldName(updateDescriptionHead.m_renames, &newClassReg);
	const hkVersionRegistry::UpdateDescription* desc = updateDescriptionHead.m_next;
	while( desc )
	{
		newClassFromOldName.registerRenames(desc->m_renames);
		desc = desc->m_next;
	}

	hkResult result = HK_SUCCESS;
	hkArray<const hkClass*> oldClasses;
	oldClassReg.getClasses(oldClasses);

	hkSort( oldClasses.begin(), oldClasses.getSize(), _compareClass );

	for( int i = 0; i < oldClasses.getSize(); ++i )
	{
		const hkClass* oldClass = oldClasses[i];
		hkUint32 oldGlobalSig = oldClass->getSignature();
		const hkVersionRegistry::ClassAction* action = actionFromOldName.getWithDefault( oldClass->getName(), HK_NULL );

		// Ignore classes with certain prefixes
		{
			const char* className = oldClass->getName();
			const int classNameLength = hkString::strLen(className);
			hkBool shouldSkip = false;

			for (int p=0; p<numIgnoredPrefixes; p++)
			{
				// class name needs to start with the prefix AND the letter after the prefix needs to be capitalized
				// (so that "hka" won't skip "hkai" classes
				const hkBool isPrefix = hkString::beginsWith( className, ignoredPrefixes[p] );
				const int prefixLength = hkString::strLen(ignoredPrefixes[p]);
				const char nextChar = ( prefixLength < classNameLength ) ? className[ prefixLength ] : 'a';
				const hkBool nextIsCap = (nextChar >= 'A' && nextChar <= 'Z' );
				if (isPrefix && nextIsCap)
				{
					shouldSkip = true;
					break;
				}
			}

			if (shouldSkip)
			{
				continue;
			}
		}

		if( oldClass->getFlags().allAreSet(hkClass::FLAGS_NOT_SERIALIZABLE) )
		{
			report.printf("%s is not serializable, can not be versioned and must be removed from the old class list.\n", oldClass->getName() );
			result = HK_FAILURE;
			continue;
		}

		// class still exists?
		const hkClass* newClass = newClassFromOldName.getClassByName( oldClass->getName() );
		if( newClass == HK_NULL
			|| (newClass && newClass->getFlags().allAreSet(hkClass::FLAGS_NOT_SERIALIZABLE))
			|| (action && (action->versionFlags & hkVersionRegistry::VERSION_REMOVED)) )
		{
			if( action == HK_NULL )
			{
				report.printf("REMOVED(%s), but no removed action\n", oldClass->getName() );
				if (! (flags & IGNORE_REMOVED) )
				{
					result = HK_FAILURE;
				}
			}
			else if( NOT(action->versionFlags & hkVersionRegistry::VERSION_REMOVED ) )
			{
				report.printf("REMOVED(%s), but action is not set to VERSION_REMOVED\n", oldClass->getName() );
				if (! (flags & IGNORE_REMOVED) )
				{
					result = HK_FAILURE;
				}
			}
			continue;
		}

		hkUint32 newGlobalSig = newClass->getSignature();
		if( containsVariants(*newClass) )
		{
			if( action == HK_NULL || !(action->versionFlags & hkVersionRegistry::VERSION_VARIANT) )
			{
				report.printf("%s (0x%x, 0x%x) has variants, but does not set VERSION_VARIANT\n", newClass->getName(), oldGlobalSig, newGlobalSig );
				result = HK_FAILURE;
				continue;
			}
		}
		if( containsHomogeneousArray(*newClass) )
		{
			if( action == HK_NULL || !(action->versionFlags & hkVersionRegistry::VERSION_HOMOGENEOUSARRAY) )
			{
				report.printf("%s (0x%x, 0x%x) has homogeneous array, but does not set VERSION_HOMOGENEOUSARRAY\n", newClass->getName(), oldGlobalSig, newGlobalSig );
				result = HK_FAILURE;
				continue;
			}
		}

		// early out if no diffs

		if( oldGlobalSig == newGlobalSig && action == HK_NULL )
		{
			continue;
		}

		// action must exist if changes exist
		if( action == HK_NULL ) 
		{
			report.printf("%s 0x%x, 0x%x MISSING ACTION\n", oldClass->getName(), oldGlobalSig, newGlobalSig );
			result = HK_FAILURE;
			continue;
		}

		hkUint32 oldLocalSig = oldClass->getSignature(hkClass::SIGNATURE_LOCAL);
		hkUint32 newLocalSig = newClass->getSignature(hkClass::SIGNATURE_LOCAL);
		const char* oldParentName = oldClass->getParent() ? oldClass->getParent()->getName() : "";
		const char* newParentName = newClass->getParent() ? newClass->getParent()->getName() : "";

		// no changes here, changes are in parent
		// Consider these cases where C is the class being checked : indicates inheritance
		// A:C -> B:C, B:C -> C, C -> B:C.
		if( oldLocalSig == newLocalSig
			// have the same inheritance/parent ?
			&& equalNames(oldParentName, newParentName, newClassFromOldName.m_renames) )
		{
			if( hkString::strCmp( action->oldClassName, oldClass->getName()) == 0 )
			{
				if( !(action->versionFlags & hkVersionRegistry::VERSION_VARIANT)
					&& !(action->versionFlags & hkVersionRegistry::VERSION_HOMOGENEOUSARRAY)
					&& equalNames( oldClass->getName(), newClass->getName(), newClassFromOldName.m_renames ) )
				{
					report.printf("%s 0x%x, 0x%x OBSOLETE ACTION\n", oldClass->getName(), oldGlobalSig, newGlobalSig );
					result = HK_FAILURE;
				}
				else if( action->oldSignature != oldGlobalSig || action->newSignature != newGlobalSig )
				{
					report.printf("%s (0x%x, 0x%x) (0x%x, 0x%x) signature mismatch\n", oldClass->getName(),
						action->oldSignature, action->newSignature,
						oldGlobalSig, newGlobalSig );
					result = HK_FAILURE;
				}
			}
			continue;
		}

		if (flags & VERBOSE)
		{
			hkArray<char> buf;
			hkOstream ostr(buf);
			hkVersionCheckingUtils::summarizeChanges( ostr, *oldClass, *newClass, true );
			report << "**" << oldClass->getName() << "\n" << buf.begin();
		}

		// changes are local - must have an entry
		if( hkString::strCmp(oldClass->getName(), action->oldClassName) != 0 )
		{
			report.printf("%s 0x%x, 0x%x has changes, but first action found for parent %s\n",
				oldClass->getName(), oldGlobalSig, newGlobalSig, action->oldClassName );
			result = HK_FAILURE;
			continue;
		}

		// is entry up to date
		if( action->oldSignature != oldGlobalSig || action->newSignature != newGlobalSig )
		{
			report.printf("%s (0x%x, 0x%x) (0x%x, 0x%x) signature mismatch\n", oldClass->getName(),
				action->oldSignature, action->newSignature,
				oldGlobalSig, newGlobalSig );
			result = HK_FAILURE;
		}

		// if parent calls func, then we should too
		if( action->versionFunc == HK_NULL )
		{
			for( const hkClass* c = oldClass->getParent(); c != HK_NULL; c = c->getParent() )
			{
				const hkVersionRegistry::ClassAction* parentAction = actionFromOldName.getWithDefault( c->getName(), HK_NULL );
				if( parentAction )
				{
					if( parentAction->versionFunc != HK_NULL )
					{
						report.printf("%s has no version func, but parent %s has\n",
							oldClass->getName(), c->getName() );
						result = HK_FAILURE;
						break;
					}
				}
				else
				{
					break;
				}
			}
		}

		// type has changed
		if( action->versionFunc == HK_NULL )
		{
			for( int memberIndex = 0; memberIndex < oldClass->getNumMembers(); ++memberIndex )
			{
				const hkClassMember& oldMem = oldClass->getMember(memberIndex);
				const hkClassMember* newMemPtr = newClass->getMemberByName( oldMem.getName() );
				if( newMemPtr )
				{
					if( !areMembersCompatible(oldMem, *newMemPtr) )
					{
						report.printf("%s::m_%s type changed but no version func\n",
							oldClass->getName(), oldMem.getName() );
						result = HK_FAILURE;
						break;
					}
					else if( oldMem.hasClass()
						&& oldMem.getType() != hkClassMember::TYPE_POINTER
						&& oldMem.getSubType() != hkClassMember::TYPE_POINTER )
					{
						for( const hkClass* oldMemClass = &oldMem.getStructClass(); oldMemClass != HK_NULL; oldMemClass = oldMemClass->getParent() )
						{
							const hkVersionRegistry::ClassAction* memClassAction = actionFromOldName.getWithDefault( oldMemClass->getName(), HK_NULL );
							if( memClassAction && memClassAction->versionFunc )
							{
								report.printf("%s::m_%s type has version func but %s has no version func\n",
									oldClass->getName(), oldMem.getName(), oldClass->getName());
								result = HK_FAILURE;
								break;
							}
						}
					}
				}
			}
		}

		// must copy if size has changed
		if( NOT( action->versionFlags & hkVersionRegistry::VERSION_COPY ) )
		{
			int oldObjectSize = oldClass->getObjectSize();
			int newObjectSize = newClass->getObjectSize();

			//XXX check size has not changed on all platforms

			if( oldObjectSize != newObjectSize )
			{
				report.printf("%s has changed size %i %i, but not set to copy\n",
					oldClass->getName(), oldObjectSize, newObjectSize );
				result = HK_FAILURE;
				continue;
			}
		}

		// if parent copies, we must too
		if( NOT( action->versionFlags & hkVersionRegistry::VERSION_COPY) )
		{
			for( const hkClass* c = oldClass->getParent(); c != HK_NULL; c = c->getParent() )
			{
				const hkVersionRegistry::ClassAction* parentAction = actionFromOldName.getWithDefault( c->getName(), HK_NULL );
				if( parentAction )
				{
					if( parentAction->versionFlags & hkVersionRegistry::VERSION_COPY )
					{
						report.printf("%s parent %s copies, so it should too.\n",
							oldClass->getName(), c->getName() );
						result = HK_FAILURE;
						break;
					}
				}
				else
				{
					break;
				}
			}
		}

		// if member copies, we must too
		if( NOT( action->versionFlags & hkVersionRegistry::VERSION_COPY) )
		{
			for( int j = 0; j < oldClass->getNumDeclaredMembers(); ++j )
			{
				const hkClassMember& m = oldClass->getDeclaredMember(j);
				if( m.hasClass()
					&& m.getType() != hkClassMember::TYPE_POINTER
					&& m.getSubType() != hkClassMember::TYPE_POINTER )
				{
					for( const hkClass* c = &m.getStructClass(); c != HK_NULL; c = c->getParent() )
					{
						const hkVersionRegistry::ClassAction* memClassAction = actionFromOldName.getWithDefault( c->getName(), HK_NULL );
						if( memClassAction )
						{
							if( memClassAction->versionFlags & hkVersionRegistry::VERSION_COPY )
							{
								report.printf("%s::m_%s type (%s) copies, so %s should too.\n",
									oldClass->getName(), m.getName(), c->getName(), oldClass->getName());
								result = HK_FAILURE;
								break;
							}
						}
					}
				}
			}
		}
	}

	// now check the table itself
	for( int i = 0; i < actionFromOldName.getJunkActions().getSize(); ++i )
	{
		report.printf("Action found, but old class '%s' is not present.\n", actionFromOldName.getJunkActions()[i]->oldClassName );
		result = HK_FAILURE;
	}

	const hkArray<const hkVersionRegistry::ClassAction*>& actions = actionFromOldName.getActions();
	for( int i = 0; i < actions.getSize(); ++i )
	{
		const hkVersionRegistry::ClassAction* action = actions[i];
		if( action->versionFlags & hkVersionRegistry::VERSION_REMOVED )
		{
			const hkClass* newClass = newClassFromOldName.getClassByName( action->oldClassName );
			if( newClass != HK_NULL && !newClass->getFlags().allAreSet(hkClass::FLAGS_NOT_SERIALIZABLE)
				// if found in global registry then
				// double check with corresponding static class name registry
				// as it may be a simple repro from newer versions,
				// the class must not be presented in the static list of classes
				&& actionFromOldName.getClassForAction(action) != HK_NULL )
			{
				report.printf("%s is marked as removed, but is still present.\n", action->oldClassName );
				result = HK_FAILURE;
			}
		}

		if( const hkClass* actionClass = oldClassReg.getClassByName( action->oldClassName ) )
		{
			for( int j = i+1; j < actions.getSize(); ++j )
			{
				const hkClass* c = oldClassReg.getClassByName( actions[j]->oldClassName );

				if( actionClass->isSuperClass(*c) )
				{
					report.printf("entry for %s is hidden by entry for %s.\n", actions[j]->oldClassName, actions[i]->oldClassName );
					result = HK_FAILURE;
				}
			}
		}
	}

	return result;
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
