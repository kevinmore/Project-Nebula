/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Compat/hkCompat.h>
#include <Common/Serialize/Data/hkDataObject.h>
#include <Common/Serialize/Version/hkVersionPatchManager.h>
#include <Common/Base/KeyCode.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>

// Registration function is at the end of the file

static void hkdDeformableBreakableShape_1_to_2(hkDataObject& obj)
{
	const hkDataArray transforms = obj["origChildTransforms"].asArray();
	hkDataArray rotations = obj["origChildTransformRotations"].asArray(); rotations.setSize(transforms.getSize());
	hkDataArray translations = obj["origChildTransformTranslations"].asArray(); translations.setSize(transforms.getSize());

	for (int i=0; i<transforms.getSize(); ++i)
	{
		const hkTransform& t = transforms[i].asTransform();

		hkQuaternion q; q.set(t.getRotation());
		hkVector4 v = t.getTranslation();

		rotations[i] = q;
		translations[i] = v;
	}
}

static void hkdShapeInstanceInfo_2_to_3(hkDataObject& obj)
{
	const hkTransform& t = obj["transform"].asTransform();

	hkQuaternion q; q.set(t.getRotation());
	hkVector4 v = t.getTranslation();

	obj["rotation"] = q;
	obj["translation"] = v;
}

static void _convertInertiaTensorToPrincipleAxis( hkMatrix3& inertia, hkRotation& principleAxisOut )
{
	principleAxisOut.setIdentity();

	for (int iterations = 5; iterations > 0; iterations--)
	{
		// find max outer diagonal element
		hkReal maxValSqrd = inertia(1,0) * inertia(1,0);
		int maxR = 1;
		int maxC = 0;
		for (int c = 0; c < 2; c++ )
		{
			hkReal in = inertia(2,c);
			in *= in;
			if ( in > maxValSqrd )
			{
				maxValSqrd = in;
				maxR = 2;
				maxC = c;
			}
		}
		if ( maxValSqrd < HK_REAL_EPSILON * HK_REAL_EPSILON)
		{
			break;
		}

		// calculate sin and cos for jacobi rotation
		hkReal si;
		hkReal co;
		{
			hkReal w = inertia( maxR, maxC );
			hkReal x = inertia( maxC, maxC );
			hkReal y = inertia( maxR, maxR );
			hkReal a = (y-x)/ (2.0f * w);
			hkReal sina = (a>0.0f) ? 1.0f : -1.0f;
			hkReal r = sina / ( hkMath::fabs(a) + hkMath::sqrt( 1 + a*a ) );
			co = hkMath::sqrtInverse( 1 + r * r );
			si = r * co;
		}
		//
		// build rotation matrix
		// and transform inertia into axis space
		//
		hkRotation rot;	rot.setIdentity();
		rot( maxC, maxC ) = co;
		rot( maxR, maxR ) = co;
		rot( maxC, maxR ) = -si;
		rot( maxR, maxC ) = si;

		hkRotation tmp; tmp.setMul( rot, inertia );

		rot( maxC, maxR ) = si;
		rot( maxR, maxC ) = -si;

		inertia.setMul( tmp, rot );

		// summing up all transformations
		tmp.setMul( principleAxisOut, rot );
		principleAxisOut = tmp;
	}

	//
	// renormalize output
	//
	principleAxisOut.getColumn(0).normalize<3>();

	principleAxisOut.getColumn(1).setCross( principleAxisOut.getColumn(2), principleAxisOut.getColumn(0) );
	principleAxisOut.getColumn(1).normalize<3>();

	principleAxisOut.getColumn(2).setCross( principleAxisOut.getColumn(0), principleAxisOut.getColumn(1) );
	principleAxisOut.getColumn(2).normalize<3>();

}

// enumerated values in version 4
namespace BreakableShape4
{
	enum ShapeType
	{
		SHAPE_TYPE_INVALID = 0, // Invalid breakable shape type
		SHAPE_TYPE_SIMPLE,      // It is a hkdBreakableShape type
		SHAPE_TYPE_COMPOUND,    // It is a hkdCompoundBreakableShape type
		SHAPE_TYPE_DEFORMABLE,  // It is a hkdDeformableBreakableShape
		SHAPE_TYPE_NUM_TYPES    // The number of shape types
	};

	/// Flags for marking breakable shapes in the filter pipeline
	enum Flags
	{
		FLAG_DO_NOT_FLATTEN_RECURSIVE_COMPOUNDS		= 1<<1,
		FLAG_SHAPE_CONTAINS_FIXED_CHILDREN			= 1<<2,
		FLAG_CHILDREN_DONT_CONNECT_TO_GRANDCHILDREN = 1<<4,		// This is set if the shapes children have no connection to the grandchildren. This allows for certain optimizations
	};
}

// enumerated values in version 5
namespace BreakableShape5
{
	enum ObjectProperties
	{
		HKD_OBJECT_PROPERTY_CONNECTION_LIST     = 0xf0001010,
		HKD_OBJECT_PROPERTY_ACTION_LIST         = 0xf0001011,
		HKD_OBJECT_PROPERTY_DYNAMIC_FRACTURE    = 0xf0001012,
		HKD_OBJECT_PROPERTY_FLAG_SET			= 0xf0001013,
		HKD_OBJECT_PROPERTY_GRAPHICS_SHAPE_NAME = 0xf0001014,
		HKD_OBJECT_PROPERTY_SHAPE_TO_CHILD_MAP  = 0xf0001015
	};

	enum TypeAndFlags
	{
		SHAPE_TYPE_OR_FLAG_INVALID                    = 0,  // Invalid breakable shape type
		SHAPE_TYPE_SIMPLE                             = 1,  // It is a hkdBreakableShape type
		SHAPE_TYPE_COMPOUND                           = 2,  // It is a hkdCompoundBreakableShape type
		SHAPE_TYPE_DEFORMABLE                         = 4,  // It is a hkdDeformableBreakableShape
		SHAPE_FLAG_DO_NOT_FLATTEN_RECURSIVE_COMPOUNDS = 8,
		SHAPE_FLAG_CONTAINS_FIXED_CHILDREN            = 16,
	};
}

static void hkdBreakableShape_4_to_5(hkDataObject& obj)
{
	const hkDataWorld* world = obj.getClass().getWorld();

	hkDataObject objectProperties = obj["objectProperties"].asObject();
	hkDataArray props = objectProperties["properties"].asArray();

	// move connections from member array to object properties
	const hkDataArray connections = obj["connections"].asArray();
	if (connections.getSize() > 0)
	{
		hkDataClass connectionListDataClass( world->findClass("hkdBreakableShapeConnectionList") );
		hkDataObject newData = world->newObject( connectionListDataClass );
		hkDataArray newConnections = newData["connections"].asArray();
		newConnections.setSize(connections.getSize());

		for (int i=0; i<connections.getSize(); ++i)
		{
			hkDataObject nC = newConnections[i].asObject();
			hkDataObject  c = connections[i].asObject();

			nC["pivotA"]             = c["pivotA"];
			nC["pivotB"]             = c["pivotB"];
			nC["separatingNormal"]   = c["separatingNormal"];
			nC["contactArea"]        = c["contactArea"];
			nC["a"]                  = c["a"]; 
			nC["b"]                  = c["b"]; 
			nC["contactAreaDetails"] = c["contactAreaDetails"];
		}

		int numProps = props.getSize();
		props.setSize(numProps + 1);

		hkDataObject newProp = props[numProps].asObject();
		newProp["key"] = (int)BreakableShape5::HKD_OBJECT_PROPERTY_CONNECTION_LIST;
		newProp["object"] = newData;
	}

	// move actions from member array to object properties
	const hkDataArray actions = obj["actions"].asArray();
	if (actions.getSize() > 0)
	{
		hkDataClass actionListDataClass( world->findClass("hkdBreakableShapeActionList") );
		hkDataObject newData = world->newObject( actionListDataClass );
		hkDataArray newActions = newData["actions"].asArray();
		newActions.setSize(actions.getSize());

		for (int i=0; i<actions.getSize(); ++i)
		{
			newActions[i] = actions[i].asObject();
		}

		int numProps = props.getSize();
		props.setSize(numProps + 1);

		hkDataObject newProp = props[numProps].asObject();
		newProp["key"] = (int)BreakableShape5::HKD_OBJECT_PROPERTY_ACTION_LIST;
		newProp["object"] = newData;
	}

	// move dynamic fracture to object properties if exist
	const hkDataObject dynFracture = obj["dynamicFracture"].asObject();
	if (dynFracture.getImplementation())
	{
		int numProps = props.getSize();
		props.setSize(numProps + 1);

		hkDataObject newProp = props[numProps].asObject();
		newProp["key"] = (int)BreakableShape5::HKD_OBJECT_PROPERTY_DYNAMIC_FRACTURE;
		newProp["object"] = dynFracture;
	}

	// convert old hkpMassProperties to new member variables
	const hkDataObject oldMassProps = obj["massProps"].asObject();
	if (oldMassProps.getImplementation())
	{
		hkVector4 massCenterMass = oldMassProps["centerOfMass"].asVector4();
		hkReal mass = oldMassProps["mass"].asReal();
		massCenterMass(3) = mass;
		obj["comAndMass"] = massCenterMass;

		hkDataArray packedValues = obj["inertiaAndValues"].asObject()["vec"].asArray();

		hkMatrix3 inertia = oldMassProps["inertiaTensor"].asMatrix3();
		hkRotation principleAxis;
		_convertInertiaTensorToPrincipleAxis( inertia, principleAxis );

		hkQuaternion prinipleAxisQ; prinipleAxisQ.set( principleAxis );
		int ax = hkVector4Util::packQuaternionIntoInt32( prinipleAxisQ.m_vec ); 
		obj["majorAxisSpace"] = ax;

		packedValues[0] = inertia(0,0);
		packedValues[1] = inertia(1,1);
		packedValues[2] = inertia(2,2);
		packedValues[3] = oldMassProps["volume"].asReal();

		packedValues[4] = obj["referenceShapeVolume"].asReal();
		packedValues[5] = obj["minDestructionRadius"].asReal();
		packedValues[6] = obj["breakingPropagationRate"].asReal();
		packedValues[7] = obj["relativeSubpieceStrength"].asReal();
	}

	// put all simple properties in a flag property set
	const hkDataObject oldFlagProperties = obj["properties"].asObject();
	if (oldFlagProperties.getImplementation())
	{
		const hkDataArray oldPropList = oldFlagProperties["properties"].asArray();

		if (oldPropList.getSize() > 0)
		{
			hkDataClass propSetDataClass( world->findClass("hkdPropertyFlagSet") );
			hkDataObject newPropFlagSet = world->newObject( propSetDataClass );

			int numProps = props.getSize();
			props.setSize(numProps + 1);

			hkDataObject newProp = props[numProps].asObject();
			newProp["key"] = (int)BreakableShape5::HKD_OBJECT_PROPERTY_FLAG_SET;
			newProp["object"] = newPropFlagSet;

			newPropFlagSet["properties"] = oldFlagProperties;
		}
	}

	// store the graphics shape name as property
	const char* graphicsName = obj["graphicsShapeName"].asString();
	if (graphicsName)
	{
		hkDataClass namePropDataClass( world->findClass("hkdStringObject") );
		hkDataObject nameProp = world->newObject( namePropDataClass );

		int numProps = props.getSize();
		props.setSize(numProps + 1);

		hkDataObject newProp = props[numProps].asObject();
		newProp["key"] = (int)BreakableShape5::HKD_OBJECT_PROPERTY_GRAPHICS_SHAPE_NAME;
		newProp["object"] = nameProp;

		nameProp["string"] = graphicsName;
	}

	// convert and combine the type and flags
	const int oldFlags = obj["flags"].asInt();
	hkUint8 newTypeAndFlags = BreakableShape5::SHAPE_TYPE_SIMPLE;

	if (obj.getClass().getImplementation() == world->findClass("hkdCompoundBreakableShape"))
	{
		newTypeAndFlags = BreakableShape5::SHAPE_TYPE_COMPOUND;
	}
	else if (obj.getClass().getImplementation() == world->findClass("hkdDeformableBreakableShape"))
	{
		newTypeAndFlags = BreakableShape5::SHAPE_TYPE_DEFORMABLE;
	}

	if (oldFlags & BreakableShape4::FLAG_DO_NOT_FLATTEN_RECURSIVE_COMPOUNDS)
	{
		newTypeAndFlags |= BreakableShape5::SHAPE_FLAG_DO_NOT_FLATTEN_RECURSIVE_COMPOUNDS;
	}
	if (oldFlags & BreakableShape4::FLAG_SHAPE_CONTAINS_FIXED_CHILDREN)
	{
		newTypeAndFlags |= BreakableShape5::SHAPE_FLAG_CONTAINS_FIXED_CHILDREN;
	}
	// BreakableShape4::FLAG_CHILDREN_DONT_CONNECT_TO_GRANDCHILDREN is deprecated

	obj["typeAndFlags"] = newTypeAndFlags;
}


// enumerated values in version 4
namespace BreakableBody4
{
	enum BreakableBodyType
	{
		BODY_TYPE_INVALID = 0, // Invalid breakable body type
		BODY_TYPE_SIMPLE,      // It is a hkdBreakableBody type
		BODY_TYPE_EMBEDDED,    // It is a hkdEmbeddedBreakableBody type
		BODY_TYPE_TEMPLATE,	   // This flags the Body as being a template. No Fracture tasks will be performed on it. This is used e.g. by the  hkdCutOutFracture and hkdVoronoiFracture.
		BODY_TYPE_NUM_TYPES    // The number of breakable body types
	};
}

// enumerated values in version 5
namespace BreakableBody5
{
	enum TypesAndFlags
	{
		BODY_TYPE_OR_FLAG_INVALID          = 0,  // Invalid breakable body type
		BODY_TYPE_SIMPLE                   = 1,  // It is a hkdBreakableBody type
		BODY_TYPE_EMBEDDED                 = 2,  // It is a hkdEmbeddedBreakableBody type
		BODY_TYPE_TEMPLATE                 = 4,  // This flags the Body as being a template. No Fracture tasks will be performed on it. This is used e.g. by the  hkdCutOutFracture and hkdVoronoiFracture.
		BODY_FLAG_ATTACH_TO_NEARBY_OBJECTS = 8,  // When set the system will find initial contact points for a newly added breakable body.
	};
}

static void hkdBreakableBody_4_to_5(hkDataObject& obj)
{
	const hkDataWorld* world = obj.getClass().getWorld();

	hkDataObject objectProperties = obj["objectProperties"].asObject();
	hkDataArray props = objectProperties["properties"].asArray();

	// convert and combine the type and flags
	const int oldType = obj["bodyType"].asInt();
	hkUint8 newTypeAndFlags = BreakableBody5::BODY_TYPE_SIMPLE;

	if (oldType == BreakableBody4::BODY_TYPE_TEMPLATE)
	{
		newTypeAndFlags = BreakableBody5::BODY_TYPE_TEMPLATE;
	}
	else if (oldType == BreakableBody4::BODY_TYPE_EMBEDDED)
	{
		newTypeAndFlags = BreakableBody5::BODY_TYPE_EMBEDDED;
	}

	if (obj["attachToNearbyObjects"].asInt() > 0)
	{
		newTypeAndFlags |= BreakableBody5::BODY_FLAG_ATTACH_TO_NEARBY_OBJECTS;
	}

	obj["bodyTypeAndFlags"] = newTypeAndFlags;

	// put all simple properties in a flag property set
	const hkDataObject oldFlagProperties = obj["properties"].asObject();
	if (oldFlagProperties.getImplementation())
	{
		const hkDataArray oldPropList = oldFlagProperties["properties"].asArray();

		if (oldPropList.getSize() > 0)
		{
			hkDataClass propSetDataClass( world->findClass("hkdPropertyFlagSet") );
			hkDataObject newPropFlagSet = world->newObject( propSetDataClass );

			int numProps = props.getSize();
			props.setSize(numProps + 1);

			hkDataObject newProp = props[numProps].asObject();
			newProp["key"] = (int)BreakableShape5::HKD_OBJECT_PROPERTY_FLAG_SET;
			newProp["object"] = newPropFlagSet;

			newPropFlagSet["properties"] = oldFlagProperties;
		}
	}

	// move shape key to child map into object properties
	const hkDataArray oldMap = obj["shapeKeyToChild"].asArray();
	if (oldMap.getSize() > 0)
	{
		hkDataClass shapeKeyMapClass( world->findClass("hkdBreakableBodyShapeKeyToChildMap") );
		hkDataObject newData = world->newObject( shapeKeyMapClass );
		hkDataArray newMap = newData["shapeKeyToChild"].asArray();
		newMap.setSize(oldMap.getSize());

		for (int i=0; i<oldMap.getSize(); ++i)
		{
			newMap[i] = oldMap[i].asObject();
		}

		int numProps = props.getSize();
		props.setSize(numProps + 1);

		hkDataObject newProp = props[numProps].asObject();
		newProp["key"] = (int)BreakableShape5::HKD_OBJECT_PROPERTY_SHAPE_TO_CHILD_MAP;
		newProp["object"] = newData;
	}
}

static void hkdDecorateFractureFaceActionShapeDecorationInfo_0_to_1(hkDataObject& obj)
{
	const hkDataWorld* world = obj.getClass().getWorld();

	// setup the global mapping table (using only uncompressed placements)
	hkDataClass globalDataClass( world->findClass("hkdDecorateFractureFaceActionGlobalDecorationData") );
	hkDataObject newGlobalData = world->newObject( globalDataClass );

	obj["sharedData"] = newGlobalData;

	const hkDataArray connDecoInfos = obj["connectionDecorations"].asArray();
	hkDataArray shapes = newGlobalData["templates"].asArray();

	// collect all decoration breakable shapes and store their index back to the placement
	for (int cdi = 0; cdi < connDecoInfos.getSize(); cdi++)
	{
		hkDataObject connDecoInfo = connDecoInfos[cdi].asObject();

		const hkDataArray placements = connDecoInfo["placements"].asArray();
		for (int di = 0; di < placements.getSize(); di++)
		{
			hkDataObject p = placements[di].asObject();

			int s = shapes.getSize();
			shapes.setSize(s+1);
			shapes[s] = p["graphicsSource"].asObject();

			hkVector4 pos = p["position"].asVector4();
			pos.setInt24W(s);
			p["position"] = pos;
		}
	}
}

static void hkdDecorateFractureFaceActionDecorationPlacement_0_to_1(hkDataObject& obj)
{
	// unpack the rotation
	hkVector4 q; 
	hkVector4Util::unPackInt32IntoQuaternion( obj["rotation"].asInt(), q );
	obj["rot"] = q;

	// position is just rename, w component has been updated by dependent patch before

	// the graphicsSource has been transferred to the global array by now, so just discard it
}

static void hkdDecorateFractureFaceAction_3_to_4(hkDataObject& obj)
{
	// proper member init
	obj["equalizeGaps"] = true;
	obj["protrusionTest"] = 0; //hkdDecorateFractureFaceAction::FAST
	obj["compressDecorations"] = true; // only used in fromFilterpipeline, so can be set safely
}

void HK_CALL registerDestructionPatches_710(hkVersionPatchManager& man)
{
#	define HK_PATCHES_FILE <Common/Compat/Patches/710/hkdPatches_710.cxx>
#	include <Common/Serialize/Version/hkVersionPatchManager.cxx>
#	undef HK_PATCHES_FILE
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
