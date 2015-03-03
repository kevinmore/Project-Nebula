/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Compat/hkCompat.h>
#include <Common/Serialize/hkSerialize.h>
#include <Common/Base/Config/hkConfigVersion.h>
#include <Common/Compat/Deprecated/Util/hkBindingClassNameRegistry.h>
#include <Common/Serialize/Util/hkBuiltinTypeRegistry.h>
#include <Common/Compat/Deprecated/Util/hkRenamedClassNameRegistry.h>
#include <Common/Serialize/Util/hkStaticClassNameRegistry.h>
#include <Common/Serialize/Util/hkStructureLayout.h>
#include <Common/Compat/Deprecated/Version/hkVersionUtilOld.h>


template<class T>
static int getNumElements(T** p)
{
	int i = 0;
	while( *p != HK_NULL )
	{
		++i;
		++p;
	}
	return i;
}

static inline void computeMemberOffsetsInplace(hkClass*const* klasses)
{
	hkStructureLayout layout;
	hkPointerMap<const hkClass*, int> done;
	hkClass*const* ci = klasses;
	while(*ci != HK_NULL)
	{
		layout.computeMemberOffsetsInplace( **ci, done );
		++ci;
	}
}

static inline int strEqual(const char* s0, const char* s1)
{
	return hkString::strCmp( s0, s1 ) == 0;
}

hkResult HK_CALL ValidatedClassNameRegistry::processClass(hkDynamicClassNameRegistry& classRegistry, const hkClass& classToProcess, void* userData)
{
	if( strEqual(classToProcess.getName(), hkClassClass.getName())
		|| strEqual(classToProcess.getName(), hkClassEnumClass.getName()) )
	{
		return HK_SUCCESS;
	}
	if( classRegistry.getClassByName(classToProcess.getName()) == HK_NULL )
	{
		classRegistry.registerClass(&classToProcess);
	}
	return HK_SUCCESS;
}

ValidatedClassNameRegistry::ValidatedClassNameRegistry(const hkClassNameRegistry* classRegistry )
{
	if( classRegistry )
	{
		merge(*classRegistry);
	}
}

void ValidatedClassNameRegistry::registerClass( const hkClass* klass, const char* name )
{
	hkDynamicClassNameRegistry::registerClass(klass, name);
	hkStringMap<hkBool32> doneClassesInOut;
	HK_ON_DEBUG(hkResult res =) validateClassRegistry(*klass, doneClassesInOut, processClass, HK_NULL);
	HK_ASSERT(0x6b9686b3, res == HK_SUCCESS);
}

hkResult ValidatedClassNameRegistry::validateClassRegistry(const hkClass& klass, hkStringMap<hkBool32>& doneClassesInOut, ClassRegistryCallbackFunc callbackFunc, void* userData)
{
	if( doneClassesInOut.hasKey(klass.getName()) )
	{
		return HK_SUCCESS;
	}
	if( callbackFunc(*this, klass, userData) == HK_FAILURE )
	{
		return HK_FAILURE;
	}

	doneClassesInOut.insert(klass.getName(), true);
	if( klass.getParent() )
	{
		if( validateClassRegistry(*klass.getParent(), doneClassesInOut, callbackFunc, userData) == HK_FAILURE )
		{
			return HK_FAILURE;
		}
	}
	for( int i = 0; i < klass.getNumDeclaredMembers(); ++i )
	{
		const hkClassMember& mem = klass.getDeclaredMember(i);
		if( mem.hasClass() )
		{
			if( validateClassRegistry(const_cast<hkClass&>(mem.getStructClass()), doneClassesInOut, callbackFunc, userData) == HK_FAILURE )
			{
				return HK_FAILURE;
			}
		}
	}
	return HK_SUCCESS;
}


hkVersionRegistry::hkVersionRegistry()
	: m_updaters( StaticLinkedUpdaters, getNumElements<const hkVersionRegistry::Updater>(StaticLinkedUpdaters), getNumElements<const hkVersionRegistry::Updater>(StaticLinkedUpdaters) )
{
}

hkVersionRegistry::~hkVersionRegistry()
{
	for (hkStringMap<hkClassNameRegistry*>::Iterator iter = m_versionToClassNameRegistryMap.getIterator(); m_versionToClassNameRegistryMap.isValid(iter); iter = m_versionToClassNameRegistryMap.getNext(iter))
	{
		hkClassNameRegistry* classRegistry = m_versionToClassNameRegistryMap.getValue(iter);
		classRegistry->removeReferenceLockUnchecked();
	}
	m_versionToClassNameRegistryMap.clear();
}

void hkVersionRegistry::registerUpdater( const Updater* updater )
{
	m_updaters.pushBack(updater);
}

hkResult hkVersionRegistry::getVersionPath( const char* fromVersion, const char* toVersion, hkArray<const Updater*>::Temp& pathOut ) const
{
	if( strEqual(fromVersion, toVersion) )
	{
		return HK_SUCCESS; // succeed trivially
	}
	pathOut.reserve(m_updaters.getSize());

	hkArray<int>::Temp nextEdge;
	nextEdge.setSize( m_updaters.getSize(), -1 );

	hkArray<int>::Temp sourceIndices;
	sourceIndices.reserve(m_updaters.getSize());
	hkArray<int>::Temp targetIndices;
	targetIndices.reserve(m_updaters.getSize());
	{
		for( int updaterIndex = 0; updaterIndex < m_updaters.getSize(); ++updaterIndex )
		{
			if( strEqual( toVersion, m_updaters[updaterIndex]->toVersion) )
			{
				// early out if we're there
				if( strEqual( fromVersion, m_updaters[updaterIndex]->fromVersion) )
				{
					pathOut.pushBack( m_updaters[updaterIndex] );
					return HK_SUCCESS;
				}

				targetIndices.pushBack( updaterIndex );
			}
			else
			{
				sourceIndices.pushBack( updaterIndex );
			}
		}
	}

	hkArray<int>::Temp nextTargetIndices; nextTargetIndices.reserve(m_updaters.getSize());
	int swaps = 0;
	while( targetIndices.getSize() )
	{
		for( int sourceIndexIndex = sourceIndices.getSize()-1; sourceIndexIndex >= 0; --sourceIndexIndex )
		{
			int sourceIndex = sourceIndices[sourceIndexIndex];

			for( int targetIndexIndex = 0; targetIndexIndex < targetIndices.getSize(); ++targetIndexIndex )
			{
				int targetIndex = targetIndices[targetIndexIndex];

				if( strEqual( m_updaters[sourceIndex]->toVersion, m_updaters[targetIndex]->fromVersion) )
				{
					nextEdge[sourceIndex] = targetIndex;
					if( strEqual( m_updaters[sourceIndex]->fromVersion, fromVersion ) )
					{
						int i = sourceIndex;
						while( i != -1 )
						{
							pathOut.pushBack(m_updaters[i]);
							i = nextEdge[i];
						}
						if( swaps & 1 ) // maintain lifo order
						{
							targetIndices.swap( nextTargetIndices );
						}
						return HK_SUCCESS;
					}
					
					nextTargetIndices.pushBack( sourceIndex );
					sourceIndices.removeAt( sourceIndexIndex );
				}
			}
		}
		targetIndices.swap( nextTargetIndices );
		swaps += 1;
		nextTargetIndices.clear();
	}
	
	if( swaps & 1 ) // maintain lifo order
	{
		targetIndices.swap( nextTargetIndices );
	}
	return HK_FAILURE;
}

static const hkStaticClassNameRegistry* getStaticClassRegistry(const char* version, hkArray<const hkStaticClassNameRegistry*>& staticClassRegistries)
{
	for( int i = staticClassRegistries.getSize()-1; i >= 0; --i )
	{
		if( strEqual(staticClassRegistries[i]->getName(), version) )
		{
			return staticClassRegistries[i];
		}
	}
	return HK_NULL;
}
const hkClassNameRegistry* hkVersionRegistry::getClassNameRegistry( const char* versionString ) const
{
	return getDynamicClassNameRegistry(versionString);
}

hkDynamicClassNameRegistry* hkVersionRegistry::getDynamicClassNameRegistry( const char* versionString ) const
{
	HK_ASSERT(0x5997db19, versionString != HK_NULL);

	hkDynamicClassNameRegistry* classRegistry = HK_NULL;
	hkResult res = m_versionToClassNameRegistryMap.get(versionString, &classRegistry);
	if (res == HK_SUCCESS)
	{
		return classRegistry;
	}
	if( !m_versionToClassNameRegistryMap.hasKey(hkVersionUtil::getCurrentVersion()) )
	{
		hkClassNameRegistry* defaultRegistry = static_cast<hkDynamicClassNameRegistry*>(hkBuiltinTypeRegistry::getInstance().getClassNameRegistry());
		HK_ASSERT(0x5997db28, defaultRegistry);
		defaultRegistry->addReference();
		m_versionToClassNameRegistryMap.insert(hkVersionUtil::getCurrentVersion(), static_cast<hkDynamicClassNameRegistry*>(defaultRegistry));
	}
	if( strEqual(versionString, hkVersionUtil::getCurrentVersion()) )
	{
		m_versionToClassNameRegistryMap.get(versionString, &classRegistry);
		return classRegistry;
	}
	HK_ASSERT(0x25bdbbe7, strEqual(versionString, hkVersionUtil::getCurrentVersion()) == false );

	hkArray<const Updater*>::Temp updaterPath;
	res = getVersionPath(versionString, hkVersionUtil::getCurrentVersion(), updaterPath);
	if( res == HK_FAILURE)
	{
		HK_WARN(0x623ef54e, "Can not find registry entry for version '" << versionString << "'.");
		return HK_NULL;
	}
	const hkClassNameRegistry* nextClassRegistry = HK_NULL;
	hkArray<const hkStaticClassNameRegistry*> staticRegArray(StaticLinkedClassRegistries, getNumElements<const hkStaticClassNameRegistry>(StaticLinkedClassRegistries), getNumElements<const hkStaticClassNameRegistry>(StaticLinkedClassRegistries));
	// find and register all versions until versionString is not found, going backward
	for( int i = updaterPath.getSize()-1; i >= 0; --i )
	{
		nextClassRegistry = m_versionToClassNameRegistryMap.getWithDefault(updaterPath[i]->toVersion, HK_NULL);
		HK_ASSERT(0x5997db28, nextClassRegistry);
		classRegistry = m_versionToClassNameRegistryMap.getWithDefault(updaterPath[i]->fromVersion, HK_NULL);
		if( !classRegistry )
		{
			// lazily create new registry
			const hkStaticClassNameRegistry* staticReg = getStaticClassRegistry(updaterPath[i]->fromVersion, staticRegArray);
			HK_ASSERT(0x5997db29, staticReg);
			hkBindingClassNameRegistry* bindingClassRegistry = new hkBindingClassNameRegistry(updaterPath[i]->desc->m_renames, nextClassRegistry);
			bindingClassRegistry->setName(staticReg->getName());
			bindingClassRegistry->merge(*staticReg);
			m_versionToClassNameRegistryMap.insert(bindingClassRegistry->getName(), bindingClassRegistry);
			classRegistry = bindingClassRegistry;
		}
		if( strEqual(updaterPath[i]->fromVersion, versionString) )
		{
			return classRegistry;
		}
	}
	// the version string is not presented in the static class
	// list we return HK_NULL (classRegistry initialized to HK_NULL)
	return classRegistry;
}

hkResult hkVersionRegistry::registerStaticClassRegistry(const hkStaticClassNameRegistry& staticReg)
{
	hkDynamicClassNameRegistry* classRegistry = getDynamicClassNameRegistry(staticReg.getName());
	HK_ASSERT3(0x5997db23, classRegistry != HK_NULL, "Class name registry is not found for version '" << staticReg.getName() << "'.");
	classRegistry->merge(ValidatedClassNameRegistry(&staticReg));
	return HK_SUCCESS;
}

hkResult hkVersionRegistry::registerUpdateDescription(hkVersionRegistry::UpdateDescription& updateDescription, const char* fromVersion, const char* toVersion)
{
	// check that the updateDescription is valid
	HK_ASSERT2(0x5997db20, fromVersion && toVersion
		&& strEqual(fromVersion, toVersion) == false, "Invalid version strings. Make sure they are valid and are not equal.");
	hkArray<const hkVersionRegistry::Updater*>::Temp pathOut;
	HK_ON_DEBUG(hkResult res =) getVersionPath(fromVersion, toVersion, pathOut);
	HK_ASSERT3(0x5997db21, res == HK_SUCCESS, "Can not find description for version path, from " << fromVersion << " to " << toVersion << ".");
	HK_ASSERT3(0x5997db22, strEqual(toVersion, pathOut[0]->toVersion), "Can not find single version path from " << fromVersion << " to " << toVersion << ".");
	hkStringMap<hkBool32> doneClasses; // list of classes checked
	hkStringMap<hkBool32> doneNewClasses; // list of classes checked
	hkDynamicClassNameRegistry* classRegistry = getDynamicClassNameRegistry(fromVersion);
	HK_ASSERT3(0x5997db23, classRegistry != HK_NULL, "Class name registry is not found for version '" << fromVersion << "'.");
	hkDynamicClassNameRegistry* classNextRegistry = getDynamicClassNameRegistry(toVersion);
	HK_ASSERT(0x5997db24, classNextRegistry);
	// set temporary renamed class registry
	hkRenamedClassNameRegistry newClassFromOldNameRegistry(HK_NULL, classNextRegistry);
	ValidatedClassNameRegistry validatedClassRegistry;
	hkVersionRegistry::UpdateDescription* desc = &updateDescription;
	while( desc )
	{
		if( desc->m_newClassRegistry )
		{
			validatedClassRegistry.merge(*desc->m_newClassRegistry);
		}
		newClassFromOldNameRegistry.registerRenames(desc->m_renames);
		desc = desc->m_next;
	}
	// register renames and new classes with global class registries
	static_cast<hkBindingClassNameRegistry*>(classRegistry)->registerRenames(newClassFromOldNameRegistry.m_renames);
	classNextRegistry->merge(validatedClassRegistry);

	// check action signatures
	desc = &updateDescription;
	while( desc )
	{
		if( desc->m_actions )
		{
			ClassAction* action = const_cast<ClassAction*>(desc->m_actions);
			for( ; action->oldClassName; ++action )
			{
				// update class signatures
				if( action->oldSignature == AUTO_SIGNATURE )
				{
					hkClass* klass = const_cast<hkClass*>(classRegistry->getClassByName(action->oldClassName));
					HK_ASSERT3(0x5997db25, klass,
						"Failed to update signature for class " << action->oldClassName << ".\n"
						"Can not find class '" << action->oldClassName << "' of old version '" << fromVersion << "'.");
					action->oldSignature = klass->getSignature();
				}
				if( action->newSignature == AUTO_SIGNATURE )
				{
					hkClass* klass = const_cast<hkClass*>(newClassFromOldNameRegistry.getClassByName(action->oldClassName));
					HK_ASSERT3(0x5997db26, klass,
						"Failed to update signature for class " << action->oldClassName << ".\n"
						"Can not find class '" << newClassFromOldNameRegistry.getRename(action->oldClassName) << "' of next version '" << toVersion << "'.");
					action->newSignature = klass->getSignature();
				}
			}
		}
		desc = desc->m_next;
	}

	// add the update description to the tail of the list
	desc = pathOut[0]->desc;
	while( desc->m_next )
	{
		desc = desc->m_next;
	}
	desc->m_next = &updateDescription;

	return HK_SUCCESS;
}

HK_SINGLETON_IMPLEMENTATION(hkVersionRegistry);

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
