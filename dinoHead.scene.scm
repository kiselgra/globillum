(let ((home (getenv "HOME")))
 (add-model "/tmp/boxes-d3.obj" :type :obj :is-base #t)
 (add-model "/share/space1/simaprus/TREX/TREX_ptx/COLOR/bc_body_trexAA.ptx"
 :type :subd :disp "/share/space1/simaprus/TREX/TREX_ptx/DISP8/bc_body_trexAA.ptx")
;  :proxy "/tmp/trex.obj")
 (add-model "/share/space1/simaprus/TREX/TREX_ptx/COLOR/hc_upteeth_trexAA.ptx"
  :type :subd :disp "/share/space1/simaprus/TREX/TREX_ptx/DISP8/hc_upteeth_trexAA.ptx")
 (subd-tess 3 3) ;use normal subdiv for 3 levels, quantization for 5 levels.
 (displacement-scale 3)
 (path-length 4)
 (path-samples 32)
 (light-samples 32)
 (use-only-those-faces "0,0:0,1:0,9:0,10:0,11:0,12:0,13:0,14:0,15:0,16:0,17:0,37:0,38:0,39:0,40:0,41:0,42:0,43:0,44:0,45:0,46:0,47:0,48:0,49:0,50:0,51:0,84:0,85:0,86:0,87:0,88:0,89:0,90:0,91:0,92:0,93:0,94:0,95:0,96:0,97:0,98:0,99:0,100:0,101:0,102:0,103:0,104:0,105:0,106:0,107:0,152:0,153:0,154:0,155:0,156:0,157:0,158:0,159:0,160:0,161:0,162:0,163:0,192:0,193:0,194:0,195:0,196:0,197:0,198:0,199:0,200:0,201:0,202:0,203:0,204:0,234:0,235:0,236:0,237:0,238:0,239:0,240:0,241:0,242:0,243:0,244:0,245:0,246:0,247:0,248:0,249:0,250:0,251:0,252:0,292:0,299:0,300:0,301:0,302:0,303:0,304:0,305:0,306:0,307:0,308:0,309:0,310:0,311:0,312:0,313:0,314:0,315:0,316:0,317:0,318:0,319:0,382:0,383:0,384:0,385:0,386:0,398:0,399:0,400:0,401:0,402:0,403:0,404:0,405:0,406:0,407:0,408:0,409:0,410:0,411:0,412:0,413:0,414:0,415:0,416:0,417:0,418:0,419:0,420:0,421:0,422:0,423:0,490:0,491:0,492:0,493:0,494:0,495:0,496:0,497:0,498:0,499:0,500:0,502:0,572:0,573:0,574:0,575:0,576:0,577:0,578:0,579:0,631:0,632:0,633:0,634:0,635:0,636:0,637:0,696:0,697:0,698:0,700:0,2169:0,2170:0,2171:0,2172:0,2208:0,2209:0,2210:0,2211:0,2212:0,2263:0,2264:0,2265:0,2266:0,2267:0,2417:0,2418:0,2422:0,2423:0,2424:0,2425:0,2426:0,2427:0,2428:0,2429:0,2430:0,2431:0,2432:0,2433:0,2434:0,2442:0,2443:0,2444:0,2482:0,2483:0,2484:0,2485:0,2486:0,2487:0,2488:0,2489:0,2490:0,2491:0,2492:0,2493:0,2494:0,2495:0,2496:0,2497:0,2521:0,2523:0,2524:0,2525:0,2526:0,2527:0,2528:0,2529:0,2530:0,2531:0,2532:0,2533:0,2534:0,2556:0,2557:0,2558:0,2559:0,2560:0,2561:0,2562:0,2563:0,2564:0,2565:0,2566:0,2588:0,2589:0,2590:0,2591:0,2592:0,2593:0,2594:0,2595:0,2596:0,2597:0,2598:0,2599:0,2600:0,2601:0,2602:0,2603:0,2604:0,2605:0,2606:0,2607:0,2608:0,2609:0,2610:0,2611:0,2638:0,2639:0,2640:0,2641:0,2642:0,2643:0,2644:0,2645:0,2646:0,2647:0,2648:0,2649:0,2650:0,2651:0,2652:0,2672:0,2673:0,2674:0,2675:0,2676:0,2677:0,2678:0,2679:0,2689:0,2690:0,2691:0,2692:1,208:1,209:1,210:1,215:1,216:1,217:1,219:1,221:1,223:1,225:1,227:1,228:1,229:1,230:1,235:1,237:1,239:1,240:1,241:1,242:1,243:1,244:1,245:1,247:1,248:1,249:1,256:1,257:1,259:1,261:1,263:1,264:1,265:1,268:1,269:1,270:1,275:1,277:1,279:1,281:1,283:1,284:1,285:1,287:1,288:1,289:1,297:1,299:1,301:1,302:1,303:1,304:1,305:1,308:1,309:1,310:1,315:1,317:1,319:1,321:1,322:1,323:1,324:1,325:1,327:1,337:1,348:1,349:1,357:1,359:")
 ;	(integrator "hybrid_area_lights")
 )
