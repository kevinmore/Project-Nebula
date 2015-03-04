/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>
#include <Common/Serialize/Util/hkBuiltinTypeRegistry.h>
#include <Common/Base/Reflection/Registry/hkDefaultClassNameRegistry.h>
#include <Common/Base/Reflection/Registry/hkTypeInfoRegistry.h>
#include <Common/Base/Reflection/Registry/hkVtableClassRegistry.h>
#include <Common/Serialize/Util/hkStaticClassNameRegistry.h>

namespace hkHavokCurrentClasses
{
	extern const char VersionString[];
	const char VersionString[] = HAVOK_SDK_VERSION_STRING;

	extern const hkStaticClassNameRegistry hkHavokDefaultClassRegistry;
	const hkStaticClassNameRegistry hkHavokDefaultClassRegistry
	(
		hkBuiltinTypeRegistry::StaticLinkedClasses,
		-1,
		VersionString
	);
} // namespace hkHavokCurrentClasses

void hkBuiltinTypeRegistry::addType( const hkTypeInfo* info, const hkClass* klass )
{
	HK_ASSERT( 0x3bcd08a0, info && klass && hkString::strCmp(info->getTypeName(), klass->getName()) == 0 );
    hkDefaultClassNameRegistry::getInstance().registerClass( klass, klass->getName() );
    hkTypeInfoRegistry::getInstance().registerTypeInfo( info );
	if( klass->getNumInterfaces() > 0 )
	{
		hkVtableClassRegistry::getInstance().registerVtable( info->getVtable(), klass );
	}
}

class hkDefaultBuiltinTypeRegistry : public hkBuiltinTypeRegistry
{
	public:
		hkDefaultBuiltinTypeRegistry()
		{
			init();
		}

		void init()
		{
			hkDefaultClassNameRegistry::getInstance().merge(hkHavokCurrentClasses::hkHavokDefaultClassRegistry);
			hkTypeInfoRegistry::getInstance().registerList( StaticLinkedTypeInfos );
			hkVtableClassRegistry::getInstance().registerList( StaticLinkedTypeInfos, StaticLinkedClasses );
		}

		virtual hkTypeInfoRegistry* getTypeInfoRegistry()
		{
            return &hkTypeInfoRegistry::getInstance();
		}

		virtual hkClassNameRegistry* getClassNameRegistry()
		{
            return &hkDefaultClassNameRegistry::getInstance();
		}

		virtual hkVtableClassRegistry* getVtableClassRegistry()
		{
            return &hkVtableClassRegistry::getInstance();
		}

		virtual void reinitialize()
		{
			hkDefaultClassNameRegistry::replaceInstance( new hkDefaultClassNameRegistry() );
			hkTypeInfoRegistry::replaceInstance( new hkTypeInfoRegistry() );
			hkVtableClassRegistry::replaceInstance( new hkVtableClassRegistry() );

			init();
		}
};

static hkReferencedObject* HK_CALL hkCreateBuiltInTypeRegistry()
{
    // Don't bother creating this until after all these have been created
	if (!&hkVtableClassRegistry::getInstance() ||
		!&hkDefaultClassNameRegistry::getInstance() ||
		!&hkTypeInfoRegistry::getInstance())
    {
        return HK_NULL;
    }
    return new hkDefaultBuiltinTypeRegistry();
}

HK_SINGLETON_CUSTOM_CALL(hkBuiltinTypeRegistry, hkCreateBuiltInTypeRegistry);

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
